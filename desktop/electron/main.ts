/**
 * Electron main process.
 *
 * Owns everything the renderer must not: global hotkeys, clipboard access, and
 * the display-media handler that supplies system-audio loopback. The renderer is
 * sandboxed and talks to this process only through the narrow preload bridge.
 */
import {
  app,
  BrowserWindow,
  clipboard,
  desktopCapturer,
  globalShortcut,
  ipcMain,
  session,
  shell,
  systemPreferences,
} from "electron";
import path from "node:path";

import { describeAudioCapabilities } from "./audio";

const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const DEV_URL = process.env.VITE_DEV_SERVER_URL ?? "http://localhost:5173";

// System-audio loopback on macOS is behind Chromium feature flags until
// Electron 39. Without them `audio: "loopback"` hands back a track that ends
// immediately, which is worse than no feature: it looks like it worked.
if (process.platform === "darwin") {
  const extra = ["MacLoopbackAudioForScreenShare", "MacSckSystemAudioLoopbackOverride"];
  const existing = app.commandLine.getSwitchValue("enable-features");
  const merged = [...new Set([...existing.split(","), ...extra])].filter(Boolean).join(",");
  app.commandLine.appendSwitch("enable-features", merged);
}

let mainWindow: BrowserWindow | null = null;

function applyCsp(): void {
  if (isDev) return; // Vite's dev client needs inline scripts; production is strict.
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        "Content-Security-Policy": [
          "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
            "img-src 'self' data:; connect-src 'self' ws://127.0.0.1:8000 http://127.0.0.1:8000",
        ],
      },
    });
  });
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 780,
    minWidth: 860,
    minHeight: 560,
    backgroundColor: "#0b0d10",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      spellcheck: false,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) void shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, url) => {
    const allowed = isDev ? DEV_URL : "file://";
    if (!url.startsWith(allowed)) event.preventDefault();
  });

  if (isDev) {
    void mainWindow.loadURL(DEV_URL);
  } else {
    void mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function registerHotkey(): void {
  const ok = globalShortcut.register("CommandOrControl+Shift+Space", () => {
    if (!mainWindow) return;
    mainWindow.show();
    mainWindow.focus();
    const selection = clipboard.readText().slice(0, 400);
    mainWindow.webContents.send("cue:hotkey", {
      windowTitle: mainWindow.getTitle(),
      selection,
      capturedAt: Date.now(),
    });
  });
  if (!ok) {
    // Another app owns the chord; surface it rather than failing silently.
    mainWindow?.webContents.send("cue:warning", {
      message: "Global shortcut Cmd/Ctrl+Shift+Space is already taken by another app.",
    });
  }
}

/**
 * Serve `getDisplayMedia({ audio: true, video: true })` from the renderer with
 * system-audio loopback instead of a picker.
 *
 * The renderer only ever needs the audio track; it stops the video track as soon
 * as the stream resolves. Requests from anywhere but our top-level frame are
 * denied, so a navigated or injected page cannot silently start a capture.
 */
function registerSystemAudio(): void {
  session.defaultSession.setDisplayMediaRequestHandler(async (request, callback) => {
    const owner = mainWindow?.webContents.mainFrame;
    const trustedOrigin = isDev ? new URL(DEV_URL).origin : "file://";
    const fromOurFrame = !request.frame || request.frame === owner;
    if (!fromOurFrame || request.securityOrigin !== trustedOrigin) {
      callback({});
      return;
    }
    try {
      const sources = await desktopCapturer.getSources({
        types: ["screen"],
        thumbnailSize: { width: 0, height: 0 },
      });
      const screen = sources[0];
      if (!screen) {
        callback({});
        return;
      }
      callback(request.audioRequested ? { video: screen, audio: "loopback" } : { video: screen });
    } catch {
      callback({});
    }
  });
}

app.whenReady().then(() => {
  applyCsp();
  registerSystemAudio();
  createWindow();
  registerHotkey();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
});

ipcMain.handle("cue:get-context", () => ({
  platform: process.platform,
  appVersion: app.getVersion(),
  windowTitle: mainWindow?.getTitle() ?? "",
  selection: clipboard.readText().slice(0, 400),
  electron: process.versions.electron,
}));

ipcMain.handle("cue:audio-capabilities", async () =>
  describeAudioCapabilities(
    process.platform,
    process.platform === "darwin"
      ? systemPreferences.getMediaAccessStatus("screen")
      : "granted",
  ),
);

ipcMain.handle("cue:open-external", async (_event, url: unknown) => {
  if (typeof url !== "string" || !url.startsWith("https://")) return { ok: false };
  await shell.openExternal(url);
  return { ok: true };
});

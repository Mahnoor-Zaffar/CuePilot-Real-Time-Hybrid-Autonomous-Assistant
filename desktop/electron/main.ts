/**
 * Electron main process.
 *
 * Owns everything the renderer must not: global hotkeys, clipboard access and
 * (in a fuller build) audio capture. The renderer is sandboxed and talks to this
 * process only through the narrow preload bridge.
 */
import { app, BrowserWindow, clipboard, globalShortcut, ipcMain, session, shell } from "electron";
import path from "node:path";

const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const DEV_URL = process.env.VITE_DEV_SERVER_URL ?? "http://localhost:5173";

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

app.whenReady().then(() => {
  applyCsp();
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

ipcMain.handle("cue:capture-audio", async () => {
  // Stub: a real build calls systemPreferences.getMediaAccessStatus / getUserMedia
  // in the renderer and pipes PCM frames over the socket.
  return {
    ok: false,
    reason: "audio capture is stubbed in the portfolio build",
    sampleRate: 16000,
    channels: 1,
  };
});

ipcMain.handle("cue:open-external", async (_event, url: unknown) => {
  if (typeof url !== "string" || !url.startsWith("https://")) return { ok: false };
  await shell.openExternal(url);
  return { ok: true };
});

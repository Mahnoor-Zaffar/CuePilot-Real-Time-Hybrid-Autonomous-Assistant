/**
 * Measure the desktop client's idle footprint.
 *
 *   npm run measure-idle
 *
 * Loads the production bundle in a hidden window, lets it settle, then reports
 * Chromium's per-process working sets plus the renderer JS heap. Electron's
 * multi-process baseline (main + GPU + utility) is reported separately so the
 * app's own cost is not confused with the runtime's.
 */
const { app, BrowserWindow } = require("electron");
const fs = require("node:fs");
const path = require("node:path");

const SETTLE_MS = 5000;
const IDLE_MS = 2000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function mb(kilobytes) {
  return Math.round((kilobytes / 1024) * 10) / 10;
}

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    width: 1180,
    height: 780,
    show: false,
    backgroundColor: "#0a0c0f",
    webPreferences: {
      preload: path.join(__dirname, "../dist-electron/preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });

  await win.loadFile(path.join(__dirname, "../dist/index.html"));
  await sleep(SETTLE_MS); // handshake + first paint + font settle

  // Ask the renderer for its own heap after a quiet window.
  const rendererHeap = await win.webContents
    .executeJavaScript(
      `(async () => {
         await new Promise((r) => setTimeout(r, ${IDLE_MS}));
         const memory = performance.memory ?? null;
         return {
           usedHeapMB: memory ? Math.round((memory.usedJSHeapSize / 1048576) * 10) / 10 : null,
           totalHeapMB: memory ? Math.round((memory.totalJSHeapSize / 1048576) * 10) / 10 : null,
           domNodes: document.getElementsByTagName("*").length,
           connected: document.querySelector(".status-pill")?.textContent?.trim() ?? null,
         };
       })()`,
    )
    .catch(() => null);

  const processes = app.getAppMetrics().map((metric) => ({
    type: metric.type,
    pid: metric.pid,
    workingSetMB: mb(metric.memory.workingSetSize),
    peakWorkingSetMB: mb(metric.memory.peakWorkingSetSize),
  }));

  const renderer = processes
    .filter((proc) => proc.type === "Tab" || proc.type === "Renderer")
    .reduce((sum, proc) => sum + proc.workingSetMB, 0);
  const total = processes.reduce((sum, proc) => sum + proc.workingSetMB, 0);

  const report = {
    measuredAt: new Date().toISOString(),
    platform: `${process.platform} ${process.arch}`,
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    processes,
    rendererWorkingSetMB: Math.round(renderer * 10) / 10,
    allProcessesWorkingSetMB: Math.round(total * 10) / 10,
    rendererHeap,
    note:
      "Electron's main/GPU/utility processes are a fixed runtime baseline; rendererWorkingSetMB is the app's own UI cost.",
  };

  process.stdout.write(`\n${JSON.stringify(report, null, 2)}\n`);
  const outDir = path.join(__dirname, "..", "bench");
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "idle.json"), `${JSON.stringify(report, null, 2)}\n`);
  win.destroy();
  app.quit();
});

app.on("window-all-closed", () => app.quit());

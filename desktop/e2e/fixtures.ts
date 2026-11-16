import {
  _electron as electron,
  test as base,
  type ElectronApplication,
  type Page,
} from "@playwright/test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const desktopDir = path.resolve(__dirname, "..");

interface Fixtures {
  electronApp: ElectronApplication;
  appWindow: Page;
}

/**
 * A fresh Electron process and a fresh Chromium profile per test.
 *
 * The profile matters: the renderer persists its session id in localStorage and
 * the gateway replays that session's transcript on reconnect. Reusing the
 * default profile would leak one test's turns into the next one's "empty app".
 */
export const test = base.extend<Fixtures>({
  electronApp: async ({}, use) => {
    const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "cuepilot-e2e-"));
    const app = await electron.launch({
      // `.` resolves through package.json `main` -> dist-electron/main.js, the
      // same entry `electron .` uses. The production renderer is loaded because
      // VITE_DEV_SERVER_URL is unset.
      args: [".", `--user-data-dir=${userDataDir}`],
      cwd: desktopDir,
    });
    try {
      await use(app);
    } finally {
      await app.close();
      fs.rmSync(userDataDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
    }
  },
  appWindow: async ({ electronApp }, use) => {
    const window = await electronApp.firstWindow();
    await window.waitForLoadState("domcontentloaded");
    await use(window);
  },
});

export { expect } from "@playwright/test";

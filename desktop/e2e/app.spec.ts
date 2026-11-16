import type { Page } from "@playwright/test";

import { expect, test } from "./fixtures";

const PROMPT = "How is TTFT measured?";

async function sendPrompt(window: Page, text: string) {
  await window.locator("textarea").fill(text);
  await window.getByRole("button", { name: "Send", exact: true }).click();
}

test.describe("cuePilot desktop", () => {
  test("connects to the gateway", async ({ appWindow }) => {
    await expect(appWindow.locator(".status-pill")).toHaveText(/connected/);
    await expect(appWindow.locator(".status-pill")).toHaveClass(/status-open/);
    // The empty state is the first thing a user sees, not a blank pane.
    await expect(appWindow.locator(".empty")).toBeVisible();
  });

  test("streams a grounded answer with retrieval sources", async ({ appWindow }) => {
    await expect(appWindow.locator(".status-pill")).toHaveText(/connected/);

    await sendPrompt(appWindow, PROMPT);

    await expect(appWindow.locator(".turn-user .user-bubble")).toHaveText(PROMPT);
    const assistant = appWindow.locator(".turn-assistant");
    await expect(assistant).toHaveCount(1);

    // The turn is genuinely streaming: a live indicator and Stop button appear
    // before the answer is complete.
    await expect(assistant.locator(".live-dot")).toBeVisible();
    await expect(appWindow.getByRole("button", { name: "Stop", exact: true })).toBeVisible();

    // ...and then the turn terminates, which is the state the user is left in.
    await expect(assistant.locator(".live-dot")).toHaveCount(0);
    await expect(appWindow.getByRole("button", { name: "Send", exact: true })).toBeVisible();
    await expect(assistant.locator(".answer")).toContainText("mock provider");

    // Retrieval is visible as provenance, not just folded into prose.
    await expect(assistant.locator(".sources")).toBeVisible();
    await expect(assistant.locator(".sources-sub")).toContainText("hybrid");
    const cards = assistant.locator(".source-card");
    expect(await cards.count()).toBeGreaterThan(0);
    await expect(cards.first().locator(".source-title")).not.toBeEmpty();
    await expect(cards.first().locator(".source-score")).toContainText(".");

    // The tool loop ran before the answer: the mock provider calls kb_search.
    await expect(assistant.locator(".tool-chip")).toHaveCount(1);
    await expect(assistant.locator(".tool-chip")).toContainText("kb_search");
    await expect(assistant.locator(".tool-chip")).toHaveClass(/tool-ok/);

    // Per-turn measurements are rendered, not merely collected.
    await expect(appWindow.locator(".stat", { hasText: "last turn" })).not.toContainText("—");
  });

  test("New session clears the transcript and reconnects", async ({ appWindow }) => {
    await expect(appWindow.locator(".status-pill")).toHaveText(/connected/);
    await sendPrompt(appWindow, PROMPT);

    await expect(appWindow.locator(".turn-user")).toHaveCount(1);
    await expect(appWindow.locator(".turn-assistant .live-dot")).toHaveCount(0);

    await appWindow.getByRole("button", { name: "New session" }).click();

    // The transcript is gone immediately; the socket reconnects behind it.
    await expect(appWindow.locator(".turn-user")).toHaveCount(0);
    await expect(appWindow.locator(".turn-assistant")).toHaveCount(0);
    await expect(appWindow.locator(".empty")).toBeVisible();
    await expect(appWindow.locator(".status-pill")).toHaveText(/connected/);
  });

  test("the global hotkey drops the selection into the draft", async ({
    appWindow,
    electronApp,
  }) => {
    await expect(appWindow.locator(".status-pill")).toHaveText(/connected/);
    const selection = "const answer = 42;";

    // The real chord cannot be pressed headlessly, so deliver the same payload
    // the main process sends when globalShortcut fires. This still crosses the
    // IPC boundary and the preload bridge into the renderer.
    await electronApp.evaluate(({ BrowserWindow }, payload) => {
      const target = BrowserWindow.getAllWindows()[0];
      target?.webContents.send("cue:hotkey", payload);
    }, { windowTitle: "E2E Window", selection, capturedAt: Date.now() });

    await expect(appWindow.locator("textarea")).toHaveValue(`${selection}\n\n`);
    await expect(appWindow.locator(".toast")).toContainText(
      `Captured ${selection.length} chars from E2E Window`,
    );
  });

  test("the preload bridge exposes context and refuses non-https URLs", async ({
    appWindow,
    electronApp,
  }) => {
    // `cue:get-context` reads the real clipboard in the main process. Save and
    // restore it so a developer's clipboard survives the suite.
    const original = await electronApp.evaluate(({ clipboard }) => clipboard.readText());
    try {
      await electronApp.evaluate(({ clipboard }) => clipboard.writeText("clipboard probe"));
      const context = await appWindow.evaluate(() => window.cue!.getContext());
      expect(context.platform).toBe(process.platform);
      expect(context.appVersion).toMatch(/^\d+\.\d+\.\d+/);
      expect(context.electron).toMatch(/^\d+\./);
      expect(context.selection).toBe("clipboard probe");
    } finally {
      await electronApp.evaluate(
        ({ clipboard }, text) => clipboard.writeText(text),
        original,
      );
    }

    // The handler is the only way the renderer can reach the shell; it must not
    // accept http:// (or anything that is not https://).
    const refused = await appWindow.evaluate(() =>
      window.cue!.openExternal("http://example.com"),
    );
    expect(refused).toEqual({ ok: false });
  });

  test("reports audio capabilities through the IPC handler", async ({ appWindow }) => {
    const caps = await appWindow.evaluate(() => window.cue!.getAudioCapabilities());

    expect(caps.platform).toBe(process.platform);
    expect(caps.sampleRate).toBe(16000);
    expect(caps.channels).toBe(1);
    expect(typeof caps.systemAudio).toBe("boolean");
    // A platform that cannot capture system audio must say why, never fail
    // silently at capture time.
    if (!caps.systemAudio) expect(caps.reason).toBeTruthy();
  });
});

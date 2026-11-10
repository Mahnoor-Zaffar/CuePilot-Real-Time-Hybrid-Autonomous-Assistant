/**
 * Smoke check for the system-audio capability policy.
 *
 * The policy decides what the renderer is told, and a wrong answer is invisible
 * at runtime: the renderer would ask for a capture that can never produce audio.
 * This runs the compiled module directly so the message the user sees is the one
 * the main process would send.
 */
const assert = require("node:assert/strict");
const { describeAudioCapabilities } = require("../dist-electron/audio.js");

const denied = describeAudioCapabilities("darwin", "denied");
assert.equal(denied.systemAudio, true);
assert.equal(denied.screenPermission, "denied");
assert.match(denied.reason, /Screen Recording permission is denied/);
assert.match(denied.reason, /System Settings/);

const granted = describeAudioCapabilities("darwin", "granted");
assert.equal(granted.systemAudio, true);
assert.equal(granted.reason, undefined);

const windows = describeAudioCapabilities("win32", "granted");
assert.equal(windows.systemAudio, true);
assert.equal(windows.reason, undefined);

const linux = describeAudioCapabilities("linux", "granted");
assert.equal(linux.systemAudio, false);
assert.match(linux.reason, /not wired up for linux/);

for (const caps of [denied, granted, windows, linux]) {
  assert.equal(caps.sampleRate, 16000);
  assert.equal(caps.channels, 1);
}

console.log("audio capability check passed: darwin denied/granted, win32, linux");

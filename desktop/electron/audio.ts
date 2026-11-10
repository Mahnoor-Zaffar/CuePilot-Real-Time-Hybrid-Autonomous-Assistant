/**
 * System-audio capability policy, kept pure so the main process and a direct
 * smoke check share exactly one implementation.
 *
 * `screenPermission` comes from `systemPreferences.getMediaAccessStatus("screen")`
 * on macOS and is `"granted"` elsewhere. A denied permission is reported as a
 * reason the renderer can show, not as a silent failure: an ended audio track
 * looks like success until the level meter stays flat.
 */
export interface AudioCapabilities {
  systemAudio: boolean;
  platform: string;
  screenPermission: string;
  sampleRate: number;
  channels: number;
  reason?: string;
}

export function describeAudioCapabilities(
  platform: string,
  screenPermission: string,
): AudioCapabilities {
  // macOS 13+ through Electron 38 needs the Chromium feature flags in main.ts;
  // Electron 39+ and Windows ship loopback directly. Linux depends on the sound
  // server, so we do not claim it here.
  const supported = platform === "darwin" || platform === "win32";
  let reason: string | undefined;
  if (!supported) {
    reason = `system audio capture is not wired up for ${platform} in this build`;
  } else if (screenPermission === "denied" || screenPermission === "restricted") {
    reason =
      "Screen Recording permission is denied. Enable it in System Settings → " +
      "Privacy & Security → Screen Recording, then restart cuePilot.";
  }
  return {
    systemAudio: supported,
    platform,
    screenPermission,
    sampleRate: 16000,
    channels: 1,
    reason,
  };
}

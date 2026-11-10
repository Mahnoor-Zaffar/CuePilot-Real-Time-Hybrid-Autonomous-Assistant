/**
 * Preload: the only surface the renderer can reach. No Node globals leak through.
 */
import { contextBridge, ipcRenderer } from "electron";

export interface CueContext {
  platform: string;
  appVersion: string;
  windowTitle: string;
  selection: string;
  electron: string;
}

export interface HotkeyPayload {
  windowTitle: string;
  selection: string;
  capturedAt: number;
}

export interface AudioCapabilities {
  systemAudio: boolean;
  platform: string;
  screenPermission: string;
  sampleRate: number;
  channels: number;
  reason?: string;
}

const api = {
  getContext: (): Promise<CueContext> => ipcRenderer.invoke("cue:get-context"),
  getAudioCapabilities: (): Promise<AudioCapabilities> =>
    ipcRenderer.invoke("cue:audio-capabilities"),
  openExternal: (url: string): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke("cue:open-external", url),
  platform: process.platform,
  onHotkey: (callback: (payload: HotkeyPayload) => void): (() => void) => {
    const listener = (_event: unknown, payload: HotkeyPayload) => callback(payload);
    ipcRenderer.on("cue:hotkey", listener);
    return () => ipcRenderer.removeListener("cue:hotkey", listener);
  },
  onWarning: (callback: (payload: { message: string }) => void): (() => void) => {
    const listener = (_event: unknown, payload: { message: string }) => callback(payload);
    ipcRenderer.on("cue:warning", listener);
    return () => ipcRenderer.removeListener("cue:warning", listener);
  },
};

contextBridge.exposeInMainWorld("cue", api);

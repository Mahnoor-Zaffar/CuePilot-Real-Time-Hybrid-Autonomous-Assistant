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

const api = {
  getContext: (): Promise<CueContext> => ipcRenderer.invoke("cue:get-context"),
  captureAudio: (): Promise<{ ok: boolean; reason?: string; sampleRate: number; channels: number }> =>
    ipcRenderer.invoke("cue:capture-audio"),
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

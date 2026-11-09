/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WS_URL?: string;
  readonly VITE_WS_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

interface CueHotkeyPayload {
  windowTitle: string;
  selection: string;
  capturedAt: number;
}

interface CueBridge {
  getContext(): Promise<{
    platform: string;
    appVersion: string;
    windowTitle: string;
    selection: string;
    electron: string;
  }>;
  captureAudio(): Promise<{ ok: boolean; reason?: string; sampleRate: number; channels: number }>;
  openExternal(url: string): Promise<{ ok: boolean }>;
  platform: string;
  onHotkey(callback: (payload: CueHotkeyPayload) => void): () => void;
  onWarning(callback: (payload: { message: string }) => void): () => void;
}

interface Window {
  cue?: CueBridge;
}

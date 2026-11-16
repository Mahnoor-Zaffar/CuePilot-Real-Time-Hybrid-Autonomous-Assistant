import { defineConfig } from "@playwright/test";
import path from "node:path";

const desktopDir = __dirname;
const serverDir = path.resolve(desktopDir, "../server");
const python = path.join(
  serverDir,
  ".venv",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);

// The production renderer is the thing under test, so the gateway must be on
// :8000: the CSP in electron/main.ts allows only that origin. Each run gets a
// fresh process, and `reuseExistingServer` is false because a developer's own
// gateway may be pointed at a real provider — the assertions below rely on the
// mock stream and the seeded corpus.
export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `"${python}" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level warning`,
    cwd: serverDir,
    url: "http://127.0.0.1:8000/api/health",
    reuseExistingServer: false,
    timeout: 90_000,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      // Mock provider + hash embedder + in-memory stores: the desktop exercises
      // the real WebSocket, retrieval, tool loop and renderer without a key,
      // Postgres or Redis.
      CUEPILOT_ENV: "test",
      CUEPILOT_LOG_LEVEL: "warning",
      CUEPILOT_LLM_PROVIDER: "mock",
      CUEPILOT_EMBEDDING_PROVIDER: "hash",
      CUEPILOT_ALLOW_DEGRADED_INFRA: "true",
      CUEPILOT_DATABASE_URL: "postgresql://cuepilot:cuepilot@127.0.0.1:1/none",
      CUEPILOT_REDIS_URL: "redis://127.0.0.1:1/0",
      CUEPILOT_INFRA_PROBE_TIMEOUT_S: "0.05",
      CUEPILOT_OTEL_ENABLED: "false",
      CUEPILOT_OTEL_EXPORTER: "none",
      CUEPILOT_SEED_ON_START: "true",
      CUEPILOT_MOCK_TTFT_MS: "10",
      // Slow enough that the streaming state is observable, still ~2 s a turn.
      CUEPILOT_MOCK_TOKEN_DELAY_MS: "15",
      // One desktop process opens a handful of sockets across tests; the
      // limiter is not what this suite is proving.
      CUEPILOT_RATE_LIMIT_ENABLED: "false",
      // No Postgres in this harness. Leaving the journal on would spool files
      // into server/data between runs.
      CUEPILOT_SESSION_JOURNAL_ENABLED: "false",
    },
  },
});

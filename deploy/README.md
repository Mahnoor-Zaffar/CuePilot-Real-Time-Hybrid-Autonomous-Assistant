# Monitoring deployment

Two independent signals, both optional:

- **Metrics** are per replica and exposed at `GET /metrics` in the Prometheus
  text format. Prometheus scrapes every replica and is the aggregation layer;
  the gateway does not merge snapshots in-process.
- **Traces** are exported over OTLP/HTTP when `CUEPILOT_OTEL_EXPORTER=otlp`.
  The same spans remain queryable in-process at `/api/traces` without a
  collector.

## Prometheus

1. Copy `deploy/prometheus/prometheus.yml` and `deploy/prometheus/alerts.yml`
   into the Prometheus configuration directory.
2. Point the `cuepilot` scrape target at every replica, not a single
   load-balanced VIP.
3. If JWT auth is enabled, `/metrics` is protected like every other route.
   Supply a bearer token via `authorization.credentials_file`; do not commit
   the token.

Alert rules cover: gateway down, provider error rate, TTFT p95, journal
durability degradation, journal drops, outbox depth/age/dead-letter, auth
rejection spikes, JWKS refresh failure, rate-limit rejections, and token
estimate drift.

The rules are checked twice in CI: `promtool` validates PromQL syntax and the
scrape config, and `server/tests/test_alerts.py` asserts that every
`cuepilot_*` series a rule names is one the gateway actually emits. The second
check is the one that catches a rule which is valid PromQL but can never fire.

The outbox gauges are refreshed on the journal replay cadence
(`CUEPILOT_SESSION_JOURNAL_SPOOL_RETRY_S`, default 30 s) and once at startup, so
a health probe does not issue a Postgres query per request. A failed probe
leaves the previous snapshot in place and increments
`session_journal_outbox_stats_errors`.

## OpenTelemetry

Run the collector with `deploy/otel-collector/config.yml` (OTLP/HTTP on 4318,
OTLP/gRPC on 4317), then set:

```sh
CUEPILOT_OTEL_EXPORTER=otlp
CUEPILOT_OTEL_ENDPOINT=http://collector:4318
```

The exporter appends `/v1/traces` to the endpoint. Replace the `debug` exporter
in the collector config with your trace backend. If the collector is
unreachable, span export fails asynchronously; the in-process trace buffer and
metrics continue to work.

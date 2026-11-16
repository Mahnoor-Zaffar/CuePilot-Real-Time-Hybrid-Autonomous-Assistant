"""Alert rules must name metrics the gateway actually emits.

`promtool check rules` validates PromQL syntax, not whether a series exists, so
a typo in a metric name produces a rule that silently never fires. This test
closes that gap by reading the metric names the runtime builds.

The scan is static: it looks for literal (or f-string) names at `metrics.incr`
and `metrics.gauge` call sites, plus the histogram registry. A name assembled
from a plain variable would escape it; the fix there is to make the name a
literal, which is also what the alert rules need in order to reference it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from app.telemetry.metrics import Metrics

REPO = Path(__file__).resolve().parents[2]
ALERTS_PATH = REPO / "deploy" / "prometheus" / "alerts.yml"
PROMETHEUS_PATH = REPO / "deploy" / "prometheus" / "prometheus.yml"
APP_DIR = REPO / "server" / "app"

# `up{job="cuepilot"}` is Prometheus' own series, so only cuepilot_* names are
# checked. A regex selector like cuepilot_provider_errors_.* yields the prefix.
_METRIC = re.compile(r"\bcuepilot_[a-z0-9_]+")
_CALL = re.compile(r"metrics\.(?:incr|gauge)\(\s*f?[\"']([^\"']+)[\"']")


def _rules() -> list[dict[str, Any]]:
    document = yaml.safe_load(ALERTS_PATH.read_text(encoding="utf-8"))
    return [rule for group in document["groups"] for rule in group["rules"]]


def _emitted() -> tuple[set[str], list[re.Pattern[str]]]:
    """(exact names, dynamic families) the gateway can render."""
    exact = {
        # Rendered by render_prometheus itself rather than by a call site.
        "cuepilot_uptime_seconds",
        "cuepilot_retrieval_cache_hit_rate",
    }
    for name in Metrics().histograms:
        exact.update(
            {
                f"cuepilot_{name}",
                f"cuepilot_{name}_bucket",
                f"cuepilot_{name}_sum",
                f"cuepilot_{name}_count",
            }
        )

    families: list[re.Pattern[str]] = []
    for path in APP_DIR.rglob("*.py"):
        for literal in _CALL.findall(path.read_text(encoding="utf-8")):
            if "{" not in literal:
                exact.add(f"cuepilot_{literal}")
                continue
            head, _, rest = literal.partition("{")
            tail = rest.partition("}")[2]
            families.append(
                re.compile(f"^cuepilot_{re.escape(head)}.+{re.escape(tail)}$")
            )
    return exact, families


def test_every_alert_metric_is_emitted() -> None:
    exact, families = _emitted()
    unknown: list[str] = []
    for rule in _rules():
        for name in _METRIC.findall(str(rule["expr"])):
            if name in exact:
                continue
            # A trailing underscore is a regex-selector prefix: any name that
            # starts with it belongs to the family.
            if name.endswith("_") and any(
                family.pattern.startswith(f"^{re.escape(name)}") for family in families
            ):
                continue
            if any(family.match(name) for family in families):
                continue
            unknown.append(f"{rule['alert']}: {name}")

    assert not unknown, "alert rules name metrics the gateway never emits:\n" + "\n".join(
        unknown
    )


def test_rules_carry_routing_metadata() -> None:
    rules = _rules()
    assert rules, "alerts.yml has no rules"

    names = [rule["alert"] for rule in rules]
    assert len(names) == len(set(names)), "alert names must be unique"

    for rule in rules:
        assert rule.get("for"), f"{rule['alert']} has no `for`, so it fires on one sample"
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["summary"]
        assert rule["annotations"]["description"]


def test_scrape_config_targets_the_metrics_route() -> None:
    config = yaml.safe_load(PROMETHEUS_PATH.read_text(encoding="utf-8"))
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert jobs["cuepilot"]["metrics_path"] == "/metrics"
    # The rule file path is a container path; mount deploy/prometheus there.
    assert config["rule_files"]

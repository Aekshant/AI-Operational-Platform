from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_helm_alerts_match_compose_alerts():
    """The chart ships a copy of the alert rules (Helm can't read outside the chart dir)."""
    a = (ROOT / "deploy/prometheus/alerts.yml").read_text()
    b = (ROOT / "deploy/helm/forgeai/files/alerts.yml").read_text()
    assert a == b, "run: cp deploy/prometheus/alerts.yml deploy/helm/forgeai/files/alerts.yml"


def test_alert_expressions_reference_real_metrics():
    from prometheus_client import REGISTRY

    from forge.gateway import metrics  # noqa: F401  (registers metric names)

    names = {m.name for m in REGISTRY.collect()}
    rules = yaml.safe_load((ROOT / "deploy/prometheus/alerts.yml").read_text())
    exprs = " ".join(r["expr"] for g in rules["groups"] for r in g["rules"])
    for metric in ("forge_ttft_seconds", "forge_requests", "forge_queue_depth", "forge_breaker_state",
                   "forge_cost_usd", "forge_routing_stale_seconds"):
        assert metric in exprs and any(n.startswith(metric) for n in names), metric

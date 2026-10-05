"""Generates deploy/grafana/dashboards/forgeai.json. Keeping the dashboard as code beats hand-edited JSON."""

import json
from pathlib import Path

P = {"type": "prometheus", "uid": "prom"}
panels, y = [], 0


def row(title):
    global y
    panels.append({"type": "row", "title": title, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "collapsed": False})
    y += 1


def ts(title, targets, unit="short", w=8, h=8, x=0, thresholds=None):
    p = {"type": "timeseries", "title": title, "datasource": P, "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 8}}, "overrides": []},
         "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
         "targets": [{"datasource": P, "expr": e, "legendFormat": l, "refId": chr(65 + i)}
                     for i, (e, l) in enumerate(targets)]}
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": [
            {"color": "green", "value": None}, {"color": "red", "value": thresholds}]}
        p["fieldConfig"]["defaults"]["custom"]["thresholdsStyle"] = {"mode": "line+area"}
    panels.append(p)


def stat(title, expr, unit, x, w=4, red=None):
    p = {"type": "stat", "title": title, "datasource": P, "gridPos": {"x": x, "y": y, "w": w, "h": 4},
         "fieldConfig": {"defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": [
             {"color": "green", "value": None}] + ([{"color": "red", "value": red}] if red is not None else [])}}},
         "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "targets": [{"datasource": P, "expr": expr, "refId": "A"}]}
    panels.append(p)


row("Production SLOs")
stat("TTFT p95", 'histogram_quantile(0.95, sum by (le) (rate(forge_ttft_seconds_bucket[1m])))', "s", 0, red=0.5)
stat("P95 latency", 'histogram_quantile(0.95, sum by (le) (rate(forge_request_duration_seconds_bucket[1m])))', "s", 4,
     red=3)
stat("Tokens/sec", 'sum(rate(forge_tokens_total{kind="completion"}[1m]))', "short", 8)
stat("Error rate", 'sum(rate(forge_requests_total{code!="200"}[1m])) / sum(rate(forge_requests_total[1m]))',
     "percentunit", 12, red=0.02)
stat("GPU utilization", "avg(DCGM_FI_DEV_GPU_UTIL)", "percent", 16)
stat("Cost / hour", "sum(rate(forge_cost_usd_total[5m])) * 3600", "currencyUSD", 20)
y += 4

row("Latency by model version")
ts("TTFT p95 by version", [('histogram_quantile(0.95, sum by (version, le) (rate(forge_ttft_seconds_bucket[1m])))',
                            "{{version}}")], "s", thresholds=0.5)
ts("E2E latency p95 by version", [(
    'histogram_quantile(0.95, sum by (version, le) (rate(forge_request_duration_seconds_bucket[1m])))', "{{version}}")],
   "s", x=8)
ts("TPOT p50 by version", [('histogram_quantile(0.5, sum by (version, le) (rate(forge_tpot_seconds_bucket[1m])))',
                            "{{version}}")], "s", x=16)
y += 8

row("Traffic & rollout")
ts("Requests/sec by version", [("sum by (version) (rate(forge_requests_total[1m]))", "{{version}}")], "reqps")
ts("Canary weight", [("max by (version) (forge_canary_weight)", "{{version}}")], "percent", x=8)
ts("Fallbacks & rejections", [("sum by (from_version, reason) (rate(forge_fallback_total[1m]))",
                               "fallback from {{from_version}} ({{reason}})"),
                              ("sum by (reason) (rate(forge_rejected_total[1m]))", "rejected: {{reason}}")],
   "reqps", x=16)
y += 8

row("Admission & resilience")
ts("Gateway queue depth / in-flight", [("max by (backend) (forge_queue_depth)", "queue {{backend}}"),
                                       ("max by (backend) (forge_inflight_requests)", "inflight {{backend}}")])
ts("Circuit breakers (0 closed, 1 half-open, 2 open)", [("max by (backend) (forge_breaker_state)", "{{backend}}")],
   x=8)
ts("Upstream errors", [("sum by (backend, kind) (rate(forge_upstream_errors_total[1m]))", "{{backend}} {{kind}}")],
   "reqps", x=16)
y += 8

row("Model servers (vLLM / DCGM)")
ts("KV cache usage", [('avg by (version) ({__name__="vllm:gpu_cache_usage_perc"})', "{{version}}")], "percentunit")
ts("Running / waiting sequences", [('sum by (version) ({__name__="vllm:num_requests_running"})', "running {{version}}"),
                                   ('sum by (version) ({__name__="vllm:num_requests_waiting"})', "waiting {{version}}")],
   x=8)
ts("GPU memory used", [("sum by (version) (DCGM_FI_DEV_FB_USED) * 1024 * 1024", "{{version}}")], "bytes", x=16)
y += 8

row("Cost")
ts("Cost/hour by version", [("sum by (version) (rate(forge_cost_usd_total[5m])) * 3600", "{{version}}")],
   "currencyUSD", w=12)
ts("Avg completion tokens per request", [(
    'sum by (version) (rate(forge_tokens_total{kind="completion"}[5m])) / sum by (version) (rate(forge_requests_total[5m]))',
    "{{version}}")], w=12, x=12)
y += 8

dash = {"uid": "forgeai", "title": "ForgeAI: Inference Platform", "schemaVersion": 39, "refresh": "5s",
        "time": {"from": "now-30m", "to": "now"}, "tags": ["forgeai"], "panels": panels,
        "annotations": {"list": [{"name": "Alerts", "datasource": P, "enable": True, "iconColor": "red",
                                  "expr": 'ALERTS{alertstate="firing"}', "titleFormat": "{{alertname}}"}]}}
out = Path(__file__).resolve().parents[1] / "deploy/grafana/dashboards/forgeai.json"
out.write_text(json.dumps(dash, indent=1))
print(f"wrote {out} ({len(panels)} panels)")

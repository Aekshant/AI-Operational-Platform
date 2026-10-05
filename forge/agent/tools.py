"""The AI SRE's tools: the only way the agent can observe or change the platform.

Every tool declares a risk level:
  * read   - side-effect free; the agent may call these freely while investigating
  * write  - changes production; never executed by the LLM directly. The agent can only
             *recommend* a write action, which the incident graph executes after a human approves.

The same registry backs the LangGraph agent, the MCP server and the control-plane API,
so there is exactly one implementation of "rollback" in the system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from forge.agent.cost import explain_cost_change
from forge.agent.prom import Loki, Prom, Tempo
from forge.config import settings
from forge.deploy import DeployController
from forge.evaluation.compare import compare
from forge.registry import Registry

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, Any]  # JSON-schema properties
    required: list[str]
    risk: str  # "read" | "write"
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": {"type": "object", "properties": self.params, "required": self.required,
                                 "additionalProperties": False}}


def _r(v: float | None, n: int = 4) -> float | None:
    return None if v is None else round(v, n)


class Toolbox:
    def __init__(self, prom: Prom | None = None, controller: DeployController | None = None,
                 registry: Registry | None = None, window: str = "2m"):
        self.prom = prom or Prom()
        self.ctl = controller or DeployController()
        self.reg = registry or self.ctl.registry
        self.window = window
        self.tempo = Tempo()
        self.loki = Loki()
        S = {"deployment": {"type": "string", "description": "Deployment name, e.g. 'qwen-farm'"}}
        self.tools: dict[str, Tool] = {t.name: t for t in [
            Tool("get_slo_status", "Current SLO compliance (TTFT p95, latency p95, error rate) per deployment.",
                 {}, [], "read", self.get_slo_status),
            Tool("get_service_health", "Gateway request rate, error rate, p95 latency and TTFT p95, per model "
                 "version, for one deployment.", S, ["deployment"], "read", self.get_service_health),
            Tool("get_model_metrics", "Model-level metrics per version: TTFT p95, TPOT p50, decode tokens/s, "
                 "average output tokens, traffic share.", S, ["deployment"], "read", self.get_model_metrics),
            Tool("get_gpu_metrics", "Per model-server: GPU utilisation, GPU memory used, KV-cache usage, "
                 "running and waiting requests.", {}, [], "read", self.get_gpu_metrics),
            Tool("get_queue_status", "Gateway admission state: queue depth, in-flight requests, rejections by "
                 "reason, circuit breaker states.", {}, [], "read", self.get_queue_status),
            Tool("get_recent_errors", "Recent upstream errors by backend and kind, fallbacks, and error log lines.",
                 {}, [], "read", self.get_recent_errors),
            Tool("get_deployment_status", "Desired rollout state (stable/canary versions, weights, revision) and "
                 "the most recent deployment events: use this to answer 'what changed?'.", S, ["deployment"],
                 "read", self.get_deployment_status),
            Tool("compare_model_versions", "Compare offline evaluation results of two registered versions "
                 "(quality, latency, cost) with the ship-gate verdict.",
                 {"model": {"type": "string"}, "base": {"type": "string"}, "candidate": {"type": "string"}},
                 ["model", "base", "candidate"], "read", self.compare_model_versions),
            Tool("search_slow_traces", "Find recent slow gateway requests in Tempo.",
                 {"min_duration_ms": {"type": "integer", "description": "Default 1000"}}, [], "read",
                 self.search_slow_traces),
            Tool("get_trace", "Span breakdown (admission, prefill, ...) of one trace.",
                 {"trace_id": {"type": "string"}}, ["trace_id"], "read", self.get_trace),
            Tool("explain_cost_change", "Attribute a change in serving cost to traffic, tokens per request, "
                 "and unit price, comparing a window with the same window at an earlier offset.",
                 {"deployment": {"type": "string"}, "window": {"type": "string", "description": "e.g. '1h'"},
                  "baseline_offset": {"type": "string", "description": "e.g. '1d'"}}, [], "read",
                 self.explain_cost),
            # ---- write tools: executed only through the approval gate ----
            Tool("rollback_model", "Abort the canary (or revert stable to the previous production version).",
                 {**S, "reason": {"type": "string"}}, ["deployment", "reason"], "write", self.rollback_model),
            Tool("pause_rollout", "Freeze the canary at its current weight.", S, ["deployment"], "write",
                 self.pause_rollout),
            Tool("set_canary_weight", "Change canary traffic percentage (0-100).",
                 {**S, "weight": {"type": "integer"}}, ["deployment", "weight"], "write", self.set_canary_weight),
            Tool("start_canary", "Start a canary of a registered version.",
                 {**S, "version": {"type": "string"}, "weight": {"type": "integer"}},
                 ["deployment", "version"], "write", self.start_canary),
            Tool("promote_canary", "Make the canary the stable version.", S, ["deployment"], "write",
                 self.promote_canary),
            Tool("scale_inference", "Set model-server replica count.", {**S, "replicas": {"type": "integer"}},
                 ["deployment", "replicas"], "write", self.scale_inference),
        ]}

    # ------------------------------------------------------------------ dispatch
    def read_tools(self) -> list[Tool]:
        return [t for t in self.tools.values() if t.risk == "read"]

    def call(self, name: str, args: dict, actor: str = "agent", idempotency_key: str | None = None) -> Any:
        tool = self.tools.get(name)
        if not tool:
            raise KeyError(f"unknown tool {name}")
        if tool.risk == "write":
            return tool.fn(**args, actor=actor, idempotency_key=idempotency_key)
        return tool.fn(**args)

    def call_json(self, name: str, args: dict) -> str:
        try:
            return json.dumps(self.call(name, args), default=str)
        except Exception as e:
            log.warning("tool %s failed: %s", name, e)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # ------------------------------------------------------------------ read tools
    def get_slo_status(self) -> dict:
        s, w = settings(), self.window
        ttft = self.prom.by(f"histogram_quantile(0.95, sum by (deployment, le) "
                            f"(rate(forge_ttft_seconds_bucket[{w}])))", "deployment")
        lat = self.prom.by(f"histogram_quantile(0.95, sum by (deployment, le) "
                           f"(rate(forge_request_duration_seconds_bucket[{w}])))", "deployment")
        err = self.prom.by(f'sum by (deployment) (rate(forge_requests_total{{code!="200"}}[{w}])) / '
                           f"sum by (deployment) (rate(forge_requests_total[{w}]))", "deployment")
        out = {}
        for d in set(ttft) | set(lat) | set(err):
            t, l_, e = ttft.get(d), lat.get(d), err.get(d) or 0.0
            out[d] = {"ttft_p95_s": _r(t), "latency_p95_s": _r(l_), "error_rate": _r(e),
                      "slo": {"ttft_p95_s": s.slo_ttft_p95_s, "latency_p95_s": s.slo_latency_p95_s,
                              "error_rate": s.slo_error_rate},
                      "violations": [n for n, v, lim in (("ttft_p95", t, s.slo_ttft_p95_s),
                                                         ("latency_p95", l_, s.slo_latency_p95_s),
                                                         ("error_rate", e, s.slo_error_rate)) if v and v > lim]}
        return out

    def get_service_health(self, deployment: str) -> dict:
        w, d = self.window, f'deployment="{deployment}"'
        rate = self.prom.by(f"sum by (version) (rate(forge_requests_total{{{d}}}[{w}]))", "version")
        err = self.prom.by(f'sum by (version) (rate(forge_requests_total{{{d},code!="200"}}[{w}])) / '
                           f"sum by (version) (rate(forge_requests_total{{{d}}}[{w}]))", "version")
        p95 = self.prom.by(f"histogram_quantile(0.95, sum by (version, le) "
                           f"(rate(forge_request_duration_seconds_bucket{{{d}}}[{w}])))", "version")
        ttft = self.prom.by(f"histogram_quantile(0.95, sum by (version, le) "
                            f"(rate(forge_ttft_seconds_bucket{{{d}}}[{w}])))", "version")
        return {v: {"rps": _r(rate.get(v), 2), "error_rate": _r(err.get(v) or 0.0), "latency_p95_s": _r(p95.get(v)),
                    "ttft_p95_s": _r(ttft.get(v))} for v in rate}

    def get_model_metrics(self, deployment: str) -> dict:
        w, d = self.window, f'deployment="{deployment}"'
        q = lambda e: self.prom.by(e, "version")  # noqa: E731
        ttft = q(f"histogram_quantile(0.95, sum by (version, le) (rate(forge_ttft_seconds_bucket{{{d}}}[{w}])))")
        tpot = q(f"histogram_quantile(0.5, sum by (version, le) (rate(forge_tpot_seconds_bucket{{{d}}}[{w}])))")
        reqs = q(f"sum by (version) (rate(forge_requests_total{{{d}}}[{w}]))")
        comp = q(f'sum by (version) (rate(forge_tokens_total{{{d},kind="completion"}}[{w}]))')
        total = sum(v or 0 for v in reqs.values()) or 1
        out = {}
        for v in reqs:
            tp = tpot.get(v)
            out[v] = {"ttft_p95_s": _r(ttft.get(v)), "tpot_p50_s": _r(tp), "decode_tokens_per_s":
                      _r(1 / tp, 1) if tp else None,
                      "avg_output_tokens": _r((comp.get(v) or 0) / reqs[v], 1) if reqs[v] else None,
                      "traffic_share": _r((reqs[v] or 0) / total, 3)}
            try:
                mv = self.reg.get(deployment, v)
                out[v]["registry_stage"] = mv["stage"]
                out[v]["offline_eval"] = {k: mv["metrics"].get(k) for k in ("accuracy", "ttft_p95_s")}
            except Exception:
                pass
        return out

    def get_gpu_metrics(self) -> dict:
        q = lambda e: self.prom.by(e, "version")  # noqa: E731
        util = q("avg by (version) (DCGM_FI_DEV_GPU_UTIL)")
        mem = q("sum by (version) (DCGM_FI_DEV_FB_USED) / 1024")
        free = q("sum by (version) (DCGM_FI_DEV_FB_FREE) / 1024")
        kv = q('avg by (version) ({__name__="vllm:gpu_cache_usage_perc"})')
        running = q('sum by (version) ({__name__="vllm:num_requests_running"})')
        waiting = q('sum by (version) ({__name__="vllm:num_requests_waiting"})')
        return {v: {"gpu_util_pct": _r(util.get(v), 1), "gpu_mem_used_gb": _r(mem.get(v), 2),
                    "gpu_mem_free_gb": _r(free.get(v), 2), "kv_cache_usage": _r(kv.get(v), 3),
                    "running": running.get(v), "waiting": waiting.get(v)} for v in util}

    def get_queue_status(self) -> dict:
        w = self.window
        return {
            "queue_depth": self.prom.by("max by (backend) (forge_queue_depth)", "backend"),
            "inflight": self.prom.by("max by (backend) (forge_inflight_requests)", "backend"),
            "rejections_per_s": self.prom.by(f"sum by (reason) (rate(forge_rejected_total[{w}]))", "reason"),
            "breakers": {k: {0: "closed", 1: "half_open", 2: "open"}.get(int(v or 0))
                         for k, v in self.prom.by("max by (backend) (forge_breaker_state)", "backend").items()},
            "routing_stale_s": self.prom.scalar("max(forge_routing_stale_seconds)"),
        }

    def get_recent_errors(self) -> dict:
        w = self.window
        out: dict[str, Any] = {
            "upstream_errors_per_s": {f"{m.get('backend')}/{m.get('kind')}": _r(v) for m, v in self.prom.query(
                f"sum by (backend, kind) (rate(forge_upstream_errors_total[{w}])) > 0")},
            "fallbacks_per_s": {f"{m.get('from_version')}/{m.get('reason')}": _r(v) for m, v in self.prom.query(
                f"sum by (from_version, reason) (rate(forge_fallback_total[{w}])) > 0")},
        }
        try:
            out["log_lines"] = self.loki.errors()
        except Exception as e:
            out["log_lines"] = f"loki unavailable: {e}"
        return out

    def get_deployment_status(self, deployment: str) -> dict:
        import time as _t

        d = self.ctl.get(deployment)
        events = self.ctl.events(deployment, limit=10)
        for e in events:
            e["minutes_ago"] = round((_t.time() - e.pop("at")) / 60, 1)
        return {"deployment": d, "recent_events": events}

    def compare_model_versions(self, model: str, base: str, candidate: str) -> dict:
        return compare(self.reg.get(model, base), self.reg.get(model, candidate))

    def search_slow_traces(self, min_duration_ms: int = 1000) -> Any:
        return self.tempo.search(min_duration_ms=min_duration_ms)

    def get_trace(self, trace_id: str) -> Any:
        return self.tempo.get(trace_id)

    def explain_cost(self, deployment: str | None = None, window: str = "1h", baseline_offset: str = "1d") -> dict:
        return explain_cost_change(self.prom, deployment, window, baseline_offset)

    # ------------------------------------------------------------------ write tools
    def rollback_model(self, deployment: str, reason: str, actor: str = "agent",
                       idempotency_key: str | None = None) -> dict:
        return self.ctl.rollback(deployment, actor=actor, reason=reason, idempotency_key=idempotency_key)

    def pause_rollout(self, deployment: str, actor: str = "agent", idempotency_key: str | None = None) -> dict:
        return self.ctl.pause(deployment, actor=actor, idempotency_key=idempotency_key)

    def set_canary_weight(self, deployment: str, weight: int, actor: str = "agent",
                          idempotency_key: str | None = None) -> dict:
        return self.ctl.set_weight(deployment, weight, actor=actor, idempotency_key=idempotency_key)

    def start_canary(self, deployment: str, version: str, weight: int = 10, actor: str = "agent",
                     idempotency_key: str | None = None) -> dict:
        return self.ctl.start_canary(deployment, version, weight, actor=actor, idempotency_key=idempotency_key)

    def promote_canary(self, deployment: str, actor: str = "agent", idempotency_key: str | None = None) -> dict:
        return self.ctl.promote(deployment, actor=actor, idempotency_key=idempotency_key)

    def scale_inference(self, deployment: str, replicas: int, actor: str = "agent",
                        idempotency_key: str | None = None) -> dict:
        return self.ctl.scale(deployment, replicas, actor=actor, idempotency_key=idempotency_key)

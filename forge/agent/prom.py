"""Small Prometheus/Tempo/Loki query clients with timeouts and retries."""

from __future__ import annotations

import math
import os
import time

import httpx

from forge.config import settings


class Prom:
    def __init__(self, url: str | None = None, timeout: float = 5.0, retries: int = 2):
        self.url = (url or settings().prometheus_url).rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _get(self, path: str, params: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = httpx.get(f"{self.url}{path}", params=params, timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
                if body.get("status") != "success":
                    raise RuntimeError(body.get("error", "prometheus query failed"))
                return body["data"]
            except (httpx.HTTPError, RuntimeError) as e:
                last = e
                time.sleep(0.2 * 2**attempt)
        raise RuntimeError(f"prometheus unavailable: {last}")

    def query(self, promql: str) -> list[tuple[dict, float]]:
        data = self._get("/api/v1/query", {"query": promql})
        out = []
        for s in data.get("result", []):
            v = float(s["value"][1])
            out.append((s["metric"], None if math.isnan(v) or math.isinf(v) else v))
        return out

    def scalar(self, promql: str, default: float | None = None) -> float | None:
        res = self.query(promql)
        return res[0][1] if res and res[0][1] is not None else default

    def by(self, promql: str, label: str) -> dict[str, float | None]:
        return {m.get(label, ""): v for m, v in self.query(promql)}


class Tempo:
    def __init__(self, url: str | None = None):
        self.url = (url or settings().tempo_url).rstrip("/")

    def search(self, service: str = "forge-gateway", min_duration_ms: int = 1000, limit: int = 5) -> list[dict]:
        r = httpx.get(f"{self.url}/api/search", params={
            "tags": f"service.name={service}", "minDuration": f"{min_duration_ms}ms", "limit": limit,
            "start": int(time.time() - 900), "end": int(time.time())}, timeout=5)
        r.raise_for_status()
        return [{"trace_id": t["traceID"], "root": t.get("rootTraceName"), "duration_ms": t.get("durationMs")}
                for t in r.json().get("traces", [])]

    def get(self, trace_id: str) -> list[dict]:
        r = httpx.get(f"{self.url}/api/traces/{trace_id}", timeout=5)
        r.raise_for_status()
        spans, seen = [], set()
        for batch in r.json().get("batches", []):
            svc = next((a["value"].get("stringValue") for a in batch.get("resource", {}).get("attributes", [])
                        if a["key"] == "service.name"), "?")
            for ss in batch.get("scopeSpans", batch.get("instrumentationLibrarySpans", [])):
                for sp in ss.get("spans", []):
                    if sp.get("spanId") in seen:
                        continue
                    seen.add(sp.get("spanId"))
                    dur = (int(sp["endTimeUnixNano"]) - int(sp["startTimeUnixNano"])) / 1e6
                    attrs = {a["key"]: next(iter(a["value"].values())) for a in sp.get("attributes", [])
                             if a["key"].startswith("forge.")}
                    spans.append({"service": svc, "name": sp["name"], "duration_ms": round(dur, 1), **attrs})
        return sorted(spans, key=lambda s: -s["duration_ms"])[:15]


class Loki:
    def __init__(self, url: str | None = None):
        self.url = (url or os.environ.get("FORGE_LOKI_URL", "http://localhost:3100")).rstrip("/")

    def errors(self, service: str = "forge-gateway", minutes: int = 15, limit: int = 20) -> list[str]:
        q = f'{{service_name="{service}"}} |~ "(?i)(error|fail|warn)"'
        r = httpx.get(f"{self.url}/loki/api/v1/query_range", params={
            "query": q, "limit": limit, "start": int((time.time() - minutes * 60) * 1e9)}, timeout=5)
        r.raise_for_status()
        lines = []
        for stream in r.json()["data"]["result"]:
            lines += [v[1][:300] for v in stream["values"]]
        return lines[:limit]

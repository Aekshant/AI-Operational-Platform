"""ForgeAI inference gateway: OpenAI-compatible, reliability-aware, fully instrumented.

Request path:
    auth -> rate limit -> guardrails -> route (canary weights) -> circuit breaker
         -> admission (bulkhead + priority queue) -> upstream (TTFT timeout, retry/fallback)
         -> stream to client while measuring TTFT / TPOT / tokens / cost
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from forge import telemetry
from forge.config import settings
from forge.gateway import metrics as m
from forge.gateway.admission import Bulkhead, Shed
from forge.gateway.auth import authenticate
from forge.gateway.breaker import STATE_VALUE, CircuitBreaker
from forge.gateway.guardrails import check_request
from forge.gateway.ratelimit import RateLimiter
from forge.gateway.retry import RetryableError, backoff
from forge.gateway.routing import Backend, RoutingTable

log = logging.getLogger("forge.gateway")
S = settings()


class Upstream:
    """An opened upstream stream whose first content chunk has already arrived."""

    def __init__(self, resp: httpx.Response, lines, first: list[dict], ttft: float):
        self.resp, self.lines, self.first, self.ttft = resp, lines, first, ttft


class Gateway:
    def __init__(self) -> None:
        self.routing = RoutingTable(S.controlplane_url, os.environ.get("FORGE_GATEWAY_KEY", "dev-admin"),
                                    S.routing_refresh_s, os.environ.get("FORGE_ROUTING_FILE"))
        self.limiter = RateLimiter(S.tenant_limits)
        self.bulkheads: dict[str, Bulkhead] = {}
        self.breakers: dict[str, CircuitBreaker] = {}
        self.client: httpx.AsyncClient | None = None

    def bulkhead(self, b: Backend) -> Bulkhead:
        if b.key not in self.bulkheads:
            self.bulkheads[b.key] = Bulkhead(S.max_inflight_per_backend, S.max_queue, S.queue_timeout_s)
        return self.bulkheads[b.key]

    def breaker(self, b: Backend) -> CircuitBreaker:
        if b.key not in self.breakers:
            self.breakers[b.key] = CircuitBreaker(S.breaker_failure_threshold, S.breaker_min_calls, S.breaker_open_s)
        return self.breakers[b.key]

    async def open_upstream(self, b: Backend, body: dict) -> Upstream:
        payload = {**body, "model": b.served_model, "stream": True, "stream_options": {"include_usage": True}}
        t0 = time.monotonic()
        try:
            req = self.client.build_request("POST", f"{b.url}/v1/chat/completions", json=payload,
                                            timeout=httpx.Timeout(S.upstream_timeout_s, connect=2.0))
            resp = await self.client.send(req, stream=True)
        except httpx.TransportError as e:
            raise RetryableError("connect", str(e)) from e
        if resp.status_code >= 500 or resp.status_code == 429:
            await resp.aclose()
            raise RetryableError("http_5xx" if resp.status_code >= 500 else "upstream_busy", str(resp.status_code))
        if resp.status_code >= 400:
            detail = (await resp.aread()).decode(errors="replace")[:500]
            await resp.aclose()
            raise HTTPException(resp.status_code, f"upstream rejected request: {detail}")
        lines = resp.aiter_lines()
        first: list[dict] = []
        try:
            while True:
                line = await asyncio.wait_for(lines.__anext__(), timeout=S.ttft_timeout_s)
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                first.append(chunk)
                if any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                    break
        except (asyncio.TimeoutError, StopAsyncIteration, httpx.TransportError) as e:
            await resp.aclose()
            raise RetryableError("ttft_timeout" if isinstance(e, asyncio.TimeoutError) else "stream_broken") from e
        return Upstream(resp, lines, first, time.monotonic() - t0)


gw = Gateway()


@asynccontextmanager
async def lifespan(app: FastAPI):
    gw.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=512, max_keepalive_connections=128))
    gw.routing.start()
    yield
    await gw.client.aclose()


app = FastAPI(title="ForgeAI Gateway", lifespan=lifespan)
tracer = telemetry.setup("forge-gateway", app)


def _error(status: int, msg: str, kind: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": kind}}, status_code=status, headers=headers)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None),
                           x_session_id: str | None = Header(default=None)):
    principal = authenticate(authorization)
    tenant = principal.tenant
    span = trace.get_current_span()
    span.set_attribute("forge.tenant", tenant)

    ok, wait = gw.limiter.check(tenant)
    if not ok:
        m.REJECTED.labels("rate_limited", tenant).inc()
        return _error(429, f"rate limit exceeded for tenant {tenant}", "rate_limited",
                      {"Retry-After": f"{max(1, round(wait))}"})

    body = await request.json()
    check_request(body, S.max_prompt_chars)
    deployment = body.get("model", "")
    client_stream = bool(body.get("stream"))

    plan = gw.routing.plan(deployment, x_session_id, S.retries + 1)
    if not plan:
        raise HTTPException(404, f"no deployment named '{deployment}'")
    primary = plan[0]
    priority = gw.limiter.priority(tenant)

    stack = AsyncExitStack()
    upstream: Upstream | None = None
    served: Backend | None = None
    last_err = "unavailable"
    t_start = time.monotonic()
    for attempt, b in enumerate(plan):
        br = gw.breaker(b)
        if not br.allow():
            last_err = "breaker_open"
            m.BREAKER.labels(b.key).set(STATE_VALUE[br.state])
            continue
        if attempt:
            await asyncio.sleep(backoff(attempt - 1))
            m.RETRIES.labels(b.key, last_err).inc()
        bh = gw.bulkhead(b)
        attempt_stack = AsyncExitStack()
        try:
            with tracer.start_as_current_span("admission") as s:
                s.set_attribute("forge.backend", b.key)
                m.QUEUE.labels(b.key).set(bh.queued + 1)
                await attempt_stack.enter_async_context(bh.admit(priority))
                m.QUEUE.labels(b.key).set(bh.queued)
                m.INFLIGHT.labels(b.key).set(bh.inflight)
            with tracer.start_as_current_span("upstream.prefill") as s:
                s.set_attribute("forge.backend", b.key)
                s.set_attribute("forge.role", b.role)
                upstream = await gw.open_upstream(b, body)
                s.set_attribute("forge.ttft_s", upstream.ttft)
            stack.push_async_exit(attempt_stack)
            served = b
            break
        except Shed as e:
            await attempt_stack.aclose()
            m.QUEUE.labels(b.key).set(bh.queued)
            m.REJECTED.labels(e.reason, tenant).inc()
            return _error(429 if e.reason == "queue_full" else 503, f"overloaded ({e.reason}); retry later",
                          "overloaded", {"Retry-After": f"{max(1, round(e.retry_after))}"})
        except RetryableError as e:
            await attempt_stack.aclose()
            last_err = e.kind
            br.record(False)
            m.BREAKER.labels(b.key).set(STATE_VALUE[br.state])
            m.UPSTREAM_ERR.labels(b.key, e.kind).inc()
            log.warning("upstream %s failed (%s), attempt %d", b.key, e.kind, attempt + 1)
        except BaseException:
            await attempt_stack.aclose()
            raise

    if upstream is None or served is None:
        m.REQUESTS.labels(deployment, primary.version, tenant, "503").inc()
        return _error(503, f"all backends failed for {deployment} (last error: {last_err})", "unavailable",
                      {"Retry-After": "2"})

    if served is not primary:
        m.FALLBACK.labels(deployment, primary.version, last_err).inc()
    route_label = served.role if served is primary else "fallback"
    stack.push_async_callback(upstream.resp.aclose)
    dv = (deployment, served.version)
    m.TTFT.labels(*dv).observe(upstream.ttft)
    headers = {"x-forge-deployment": deployment, "x-forge-version": served.version, "x-forge-route": route_label}
    if tid := telemetry.current_trace_id():
        headers["x-trace-id"] = tid

    async def relay():
        """Yields upstream chunks; records metrics and releases capacity when done."""
        usage: dict = {}
        n_chunks, t_first, ok = 0, time.monotonic(), False
        try:
            for chunk in upstream.first:
                n_chunks += 1
                yield chunk
            async for line in upstream.lines:
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                n_chunks += 1
                yield chunk
            ok = True
        except (httpx.TransportError, asyncio.TimeoutError) as e:
            m.UPSTREAM_ERR.labels(served.key, "midstream").inc()
            log.error("stream from %s broke after first token: %s", served.key, e)
            yield {"error": {"message": "upstream stream interrupted", "type": "upstream_error"}}
        finally:
            elapsed = time.monotonic() - t_start
            gw.breaker(served).record(ok)
            m.BREAKER.labels(served.key).set(STATE_VALUE[gw.breaker(served).state])
            m.DURATION.labels(*dv).observe(elapsed)
            completion = usage.get("completion_tokens", n_chunks)
            prompt_tok = usage.get("prompt_tokens", 0)
            if completion > 1:
                m.TPOT.labels(*dv).observe((time.monotonic() - t_first) / (completion - 1))
            m.TOKENS.labels(*dv, tenant, "prompt").inc(prompt_tok)
            m.TOKENS.labels(*dv, tenant, "completion").inc(completion)
            m.COST.labels(*dv, tenant).inc((prompt_tok + completion) / 1000 * served.price_per_1k_tokens)
            m.REQUESTS.labels(*dv, tenant, "200" if ok else "502").inc()
            await stack.aclose()
            m.INFLIGHT.labels(served.key).set(gw.bulkhead(served).inflight)

    if client_stream:
        async def sse():
            async for chunk in relay():
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

    text, usage, finish, rid = [], {}, "stop", ""
    async for chunk in relay():
        if "error" in chunk:
            return _error(502, chunk["error"]["message"], "upstream_error", headers)
        rid = chunk.get("id", rid)
        usage = chunk.get("usage") or usage
        for c in chunk.get("choices", []):
            text.append(c.get("delta", {}).get("content") or "")
            finish = c.get("finish_reason") or finish
    return JSONResponse({
        "id": rid, "object": "chat.completion", "created": int(time.time()), "model": deployment,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text).strip()},
                     "finish_reason": finish}],
        "usage": usage,
    }, headers=headers)


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    authenticate(authorization)
    return {"object": "list", "data": [
        {"id": name, "object": "model", "owned_by": "forgeai",
         "versions": [{"version": b.version, "role": b.role, "weight": b.weight} for b in r.backends]}
        for name, r in gw.routing.routes.items()]}


@app.get("/healthz")
async def healthz():
    stale = time.time() - gw.routing.last_ok if gw.routing.last_ok else None
    return {"status": "ok" if gw.routing.routes else "degraded", "routes": list(gw.routing.routes),
            "routing_stale_s": stale,
            "breakers": {k: b.state for k, b in gw.breakers.items()}}


@app.get("/metrics")
async def prom_metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

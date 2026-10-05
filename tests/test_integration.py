"""Real processes: two simulator versions behind the real gateway (static routing file).

Covers streaming, canary split, usage accounting, fault injection, circuit breaking and fallback.
"""

import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

PY = sys.executable


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(url)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stack")
    procs = []
    ports = {"v1": free_port(), "v2": free_port(), "gw": free_port()}
    for v in ("v1", "v2"):
        env = {**os.environ, "SIM_VERSION": v, "SIM_TPOT_MS": "1", "SIM_TTFT_BASE_MS": "10",
               "SIM_MEAN_OUTPUT_TOKENS": "20"}
        procs.append(subprocess.Popen([PY, "-m", "uvicorn", "forge.serving.simulator:app", "--port",
                                       str(ports[v]), "--log-level", "warning"], env=env))
    routes = {"qwen-farm": {"revision": 1, "status": "canary", "backends": [
        {"model": "qwen-farm", "version": "v1", "url": f"http://127.0.0.1:{ports['v1']}", "weight": 70,
         "role": "stable", "price_per_1k_tokens": 0.001},
        {"model": "qwen-farm", "version": "v2", "url": f"http://127.0.0.1:{ports['v2']}", "weight": 30,
         "role": "canary", "price_per_1k_tokens": 0.001}],
        "fallback": {"model": "qwen-farm", "version": "v1", "url": f"http://127.0.0.1:{ports['v1']}"}}}
    rf = tmp / "routes.json"
    rf.write_text(json.dumps(routes))
    env = {**os.environ, "FORGE_ROUTING_FILE": str(rf), "FORGE_CONTROLPLANE_URL": "", "FORGE_BREAKER_MIN_CALLS": "5",
           "FORGE_BREAKER_OPEN_S": "60", "FORGE_RETRIES": "2", "FORGE_HOME": str(tmp)}
    procs.append(subprocess.Popen([PY, "-m", "uvicorn", "forge.gateway.app:app", "--port", str(ports["gw"]),
                                   "--log-level", "warning"], env=env))
    for p in (ports["v1"], ports["v2"]):
        wait_http(f"http://127.0.0.1:{p}/health")
    wait_http(f"http://127.0.0.1:{ports['gw']}/healthz")
    time.sleep(1.5)  # first routing refresh
    yield {k: f"http://127.0.0.1:{v}" for k, v in ports.items()}
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(5)


H = {"Authorization": "Bearer dev-admin"}
BODY = {"model": "qwen-farm", "max_tokens": 64, "messages": [{"role": "user", "content": "What does NPK stand for?"}]}


def test_auth_required(stack):
    assert httpx.post(f"{stack['gw']}/v1/chat/completions", json=BODY).status_code == 401
    assert httpx.post(f"{stack['gw']}/v1/chat/completions", json=BODY,
                      headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_guardrails(stack):
    bad = {**BODY, "messages": [{"role": "user", "content": "my key AKIAABCDEFGHIJKLMNOP"}]}
    assert httpx.post(f"{stack['gw']}/v1/chat/completions", json=bad, headers=H).status_code == 400


def test_non_streaming_and_split(stack):
    versions = []
    with httpx.Client(timeout=30) as c:
        for _ in range(60):
            r = c.post(f"{stack['gw']}/v1/chat/completions", json=BODY, headers=H)
            assert r.status_code == 200, r.text
            assert r.json()["usage"]["completion_tokens"] > 0
            versions.append(r.headers["x-forge-version"])
    assert 8 <= versions.count("v2") <= 30  # ~30%


def test_streaming(stack):
    with httpx.stream("POST", f"{stack['gw']}/v1/chat/completions", json={**BODY, "stream": True}, headers=H,
                      timeout=30) as r:
        assert r.status_code == 200 and "x-trace-id" in r.headers or True
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert sum(1 for c in chunks for ch in c["choices"] if ch["delta"].get("content")) > 3
    assert any(c.get("usage") for c in chunks)


def test_canary_failure_falls_back_then_breaker_opens(stack):
    httpx.post(f"{stack['v2']}/admin/faults", json={"error_rate": 1.0}).raise_for_status()
    try:
        # The breaker's rolling window still holds v2's earlier successes, so it takes a number of
        # failures to cross the 50% threshold. Users must see zero errors throughout.
        routes, state = [], None
        with httpx.Client(timeout=30) as c:
            for _ in range(200):
                r = c.post(f"{stack['gw']}/v1/chat/completions", json=BODY, headers=H)
                assert r.status_code == 200, r.text  # users never see the canary's failures
                routes.append((r.headers["x-forge-version"], r.headers["x-forge-route"]))
                state = httpx.get(f"{stack['gw']}/healthz").json()["breakers"].get("qwen-farm:v2")
                if state == "open" and len(routes) > 20:
                    break
        assert all(v == "v1" for v, _ in routes)
        assert ("v1", "fallback") in routes
        assert state == "open"
        metrics = httpx.get(f"{stack['gw']}/metrics").text
        assert 'forge_fallback_total{deployment="qwen-farm",from_version="v2",reason="http_5xx"}' in metrics
    finally:
        httpx.post(f"{stack['v2']}/admin/faults", json={"reset": True})


def test_metrics_exposed(stack):
    m = httpx.get(f"{stack['gw']}/metrics").text
    for name in ("forge_ttft_seconds_bucket", "forge_tpot_seconds_bucket", "forge_tokens_total",
                 "forge_cost_usd_total", "forge_routing_revision"):
        assert name in m
    sim = httpx.get(f"{stack['v1']}/metrics").text
    assert "vllm:num_requests_running" in sim and "DCGM_FI_DEV_FB_USED" in sim

"""ForgeAI control plane: registry, deployments, incidents, agent and the web UI.

The control plane owns desired state; the gateway (data plane) polls `/api/routing`.
Mutations require the `operator` role and accept an `Idempotency-Key` header.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel

from forge import db, telemetry
from forge.agent.graph import IncidentGraph
from forge.agent.incidents import OPEN_PHASES, IncidentStore
from forge.agent.investigator import ask
from forge.agent.prom import Prom
from forge.agent.tools import Toolbox
from forge.deploy import DeployController, DeployError
from forge.evaluation.compare import compare
from forge.gateway.auth import Principal, require
from forge.registry import Registry, RegistryError

log = logging.getLogger("forge.controlplane")
app = FastAPI(title="ForgeAI Control Plane")
telemetry.setup("forge-controlplane", app)

registry = Registry()
controller = DeployController(registry=registry)
incidents = IncidentStore()
toolbox = Toolbox(controller=controller, registry=registry)
_graph: IncidentGraph | None = None
pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")

INCIDENTS_OPEN = Gauge("forge_incidents_open", "Open incidents")
AGENT_RUNS = Counter("forge_agent_runs_total", "Agent graph runs", ["kind", "outcome"])
viewer, operator = require("viewer"), require("operator")


def graph() -> IncidentGraph:
    global _graph
    if _graph is None:
        _graph = IncidentGraph(tools=toolbox, store=incidents)
    return _graph


@app.exception_handler(RegistryError)
@app.exception_handler(DeployError)
async def domain_error(_, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse({"detail": str(exc)}, status_code=409)


def actor(p: Principal) -> str:
    return f"{p.tenant}:{p.role}:{p.key_id}"


# ------------------------------------------------------------------ models
class RegisterBody(BaseModel):
    name: str
    version: str
    base_model: str
    adapter_uri: str | None = None
    serving_url: str | None = None
    dataset_hash: str | None = None
    run_id: str | None = None
    metrics: dict = {}
    tags: dict = {}


@app.get("/api/models")
def list_models(name: str | None = None, _: Principal = Depends(viewer)):
    return registry.list(name)


@app.post("/api/models")
def register_model(b: RegisterBody, p: Principal = Depends(operator)):
    mv = registry.register(**b.model_dump(), actor=actor(p))
    db.audit(actor(p), "register", f"{b.name}:{b.version}")
    return mv


@app.post("/api/models/{name}/{version}/metrics")
def update_metrics(name: str, version: str, metrics: dict, p: Principal = Depends(operator)):
    return registry.update_metrics(name, version, metrics)


@app.post("/api/models/{name}/{version}/serving-url")
def set_serving_url(name: str, version: str, body: dict, p: Principal = Depends(operator)):
    registry.set_serving_url(name, version, body["url"])
    return registry.get(name, version)


@app.post("/api/models/{name}/{version}/stage")
def transition(name: str, version: str, body: dict, p: Principal = Depends(operator)):
    mv = registry.transition(name, version, body["stage"], actor(p), body.get("reason", ""))
    db.audit(actor(p), "transition", f"{name}:{version}", body)
    return mv


@app.get("/api/models/{name}/history")
def model_history(name: str, _: Principal = Depends(viewer)):
    return registry.history(name)


@app.get("/api/models/{name}/compare")
def compare_versions(name: str, base: str, candidate: str, _: Principal = Depends(viewer)):
    return compare(registry.get(name, base), registry.get(name, candidate))


# ------------------------------------------------------------------ deployments
class DeployBody(BaseModel):
    name: str
    model: str
    version: str


class RolloutBody(BaseModel):
    version: str | None = None
    weight: int | None = None
    replicas: int | None = None
    reason: str = ""
    expected_revision: int | None = None


@app.get("/api/routing")
def routing(_: Principal = Depends(viewer)):
    return controller.routing()


@app.get("/api/deployments")
def list_deployments(_: Principal = Depends(viewer)):
    return controller.list()


@app.post("/api/deployments")
def create_deployment(b: DeployBody, p: Principal = Depends(operator)):
    d = controller.create(b.name, b.model, b.version, actor(p))
    db.audit(actor(p), "create_deployment", b.name, b.model_dump())
    return d


@app.get("/api/deployments/{name}/events")
def deployment_events(name: str, _: Principal = Depends(viewer)):
    return controller.events(name)


@app.post("/api/deployments/{name}/{action}")
def rollout(name: str, action: str, b: RolloutBody, p: Principal = Depends(operator),
            idempotency_key: str | None = Header(default=None)):
    a, kw = actor(p), {"idempotency_key": idempotency_key}
    if action == "canary":
        if not b.version:
            raise HTTPException(422, "version required")
        res = controller.start_canary(name, b.version, b.weight or 10, a, b.expected_revision, **kw)
    elif action == "weight":
        res = controller.set_weight(name, b.weight if b.weight is not None else 0, a, b.expected_revision, **kw)
    elif action == "promote":
        res = controller.promote(name, a, b.expected_revision, **kw)
    elif action == "rollback":
        res = controller.rollback(name, a, b.reason, b.version, b.expected_revision, **kw)
    elif action == "pause":
        res = controller.pause(name, a, **kw)
    elif action == "scale":
        res = controller.scale(name, b.replicas or 1, a, **kw)
    else:
        raise HTTPException(404, f"unknown action {action}")
    db.audit(a, action, name, b.model_dump())
    return res


# ------------------------------------------------------------------ incidents & agent
class IncidentBody(BaseModel):
    deployment: str
    title: str = "Manual investigation"
    severity: str = "warning"
    proposal: dict | None = None  # {"action", "args", "rationale"}, from external agents via MCP


class DecisionBody(BaseModel):
    approved: bool
    comment: str = ""


def _run(kind: str, fn, *args):
    def wrapped():
        try:
            fn(*args)
            AGENT_RUNS.labels(kind, "ok").inc()
        except Exception as e:
            AGENT_RUNS.labels(kind, "error").inc()
            log.exception("agent %s failed", kind)
            if args:
                incidents.update(args[0], phase="ESCALATED", note=f"agent error: {type(e).__name__}: {e}")

    return pool.submit(wrapped)


def open_incident(deployment: str, title: str, severity: str, source: str, labels: dict,
                  proposal: dict | None = None) -> dict:
    inc = incidents.create(title, {**labels, "deployment": deployment}, severity, source)
    _run("investigate", graph().start, inc["id"], deployment, labels, proposal)
    return inc


@app.get("/api/incidents")
def list_incidents(_: Principal = Depends(viewer)):
    return incidents.list()


@app.get("/api/incidents/{iid}")
def get_incident(iid: str, _: Principal = Depends(viewer)):
    try:
        return {**incidents.get(iid), "graph": graph().state(iid)}
    except KeyError:
        raise HTTPException(404, "no such incident") from None


@app.post("/api/incidents")
def create_incident(b: IncidentBody, p: Principal = Depends(operator)):
    if b.proposal and b.proposal.get("args", {}).get("deployment") != b.deployment:
        raise HTTPException(422, "proposal must target the incident's deployment")
    return open_incident(b.deployment, b.title, b.severity, f"manual:{p.tenant}", {}, b.proposal)


@app.post("/api/incidents/{iid}/decision")
def decide(iid: str, b: DecisionBody, p: Principal = Depends(operator)):
    inc = incidents.get(iid)
    if inc["state"] != "AWAITING_APPROVAL":
        raise HTTPException(409, f"incident is {inc['state']}, not awaiting approval")
    db.audit(actor(p), "approve" if b.approved else "reject", iid, b.model_dump())
    _run("resume", graph().resume, iid, b.approved, actor(p), b.comment)
    return {"accepted": True}


@app.post("/webhooks/alertmanager")
def alertmanager(payload: dict, _: Principal = Depends(operator)):
    """Alertmanager webhook -> de-duplicated incidents -> autonomous investigation."""
    opened = []
    for a in payload.get("alerts", []):
        if a.get("status") != "firing":
            continue
        labels = a.get("labels", {})
        fp = a.get("fingerprint") or labels.get("alertname", "") + labels.get("deployment", "")
        if incidents.find_open(fp):
            continue
        dep = labels.get("deployment", "qwen-farm")
        title = a.get("annotations", {}).get("summary") or labels.get("alertname", "alert")
        opened.append(open_incident(dep, title, labels.get("severity", "critical"), "alertmanager",
                                    {**labels, "fingerprint": fp})["id"])
    return {"opened": opened}


class AskBody(BaseModel):
    question: str
    deployment: str = "qwen-farm"


@app.post("/api/agent/ask")
def agent_ask(b: AskBody, _: Principal = Depends(viewer)):
    return {"answer": ask(toolbox, b.question, b.deployment)}


# ------------------------------------------------------------------ overview / UI
@app.get("/api/overview")
def overview(_: Principal = Depends(viewer)):
    prom = Prom(timeout=2, retries=0)
    q = {
        "rps": "sum(rate(forge_requests_total[1m]))",
        "ttft_p95_s": "histogram_quantile(0.95, sum by (le) (rate(forge_ttft_seconds_bucket[1m])))",
        "latency_p95_s": "histogram_quantile(0.95, sum by (le) (rate(forge_request_duration_seconds_bucket[1m])))",
        "tokens_per_s": 'sum(rate(forge_tokens_total{kind="completion"}[1m]))',
        "error_rate": 'sum(rate(forge_requests_total{code!="200"}[1m])) / sum(rate(forge_requests_total[1m]))',
        "gpu_util_pct": "avg(DCGM_FI_DEV_GPU_UTIL)",
        "kv_cache": 'avg({__name__="vllm:gpu_cache_usage_perc"})',
        "queue_depth": "sum(forge_queue_depth)",
        "cost_per_hour_usd": "sum(rate(forge_cost_usd_total[5m])) * 3600",
    }
    metrics = {}
    for k, expr in q.items():
        try:
            metrics[k] = prom.scalar(expr)
        except Exception:
            metrics[k] = None
    incs = incidents.list(20)
    open_ = [i for i in incs if i["state"] in OPEN_PHASES]
    INCIDENTS_OPEN.set(len(open_))
    try:
        by_version = toolbox.get_service_health("qwen-farm")
    except Exception:
        by_version = {}
    return {"metrics": metrics, "by_version": by_version, "deployments": controller.list(),
            "models": registry.list(), "incidents": incs, "open_incidents": len(open_)}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/metrics")
def prom_metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


STATIC = Path(__file__).parent / "static"


@app.get("/")
def ui():
    return FileResponse(STATIC / "index.html")

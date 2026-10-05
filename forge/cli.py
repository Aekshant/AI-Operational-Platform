"""`forge`: the ForgeAI command line.

Platform-state commands go through the control-plane API, so RBAC and the audit log apply to
humans and agents alike. Training and evaluation run where the CLI runs (a GPU box, a CI job).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Optional

import httpx
import typer
import yaml
from rich.console import Console
from rich.table import Table

from forge.config import settings

app = typer.Typer(help="ForgeAI: train, evaluate, register, deploy and operate small language models.",
                  no_args_is_help=True)
model_app = typer.Typer(help="Model lifecycle: train → evaluate → register → deploy → rollback.", no_args_is_help=True)
agent_app = typer.Typer(help="The AI SRE.", no_args_is_help=True)
incident_app = typer.Typer(help="Incidents and approvals.", no_args_is_help=True)
platform_app = typer.Typer(help="Platform status.", no_args_is_help=True)
demo_app = typer.Typer(help="End-to-end demo scenarios against the simulator stack.", no_args_is_help=True)
serve_app = typer.Typer(help="Run a platform service.", no_args_is_help=True)
for name, sub in (("model", model_app), ("agent", agent_app), ("incident", incident_app),
                  ("platform", platform_app), ("demo", demo_app), ("serve", serve_app)):
    app.add_typer(sub, name=name)

con = Console()


class API:
    def __init__(self) -> None:
        self.base = settings().controlplane_url.rstrip("/")
        self.key = os.environ.get("FORGE_API_KEY", "dev-admin")

    def __call__(self, method: str, path: str, json_body: dict | None = None, idem: bool = False):
        headers = {"Authorization": f"Bearer {self.key}"}
        if idem:
            headers["Idempotency-Key"] = str(uuid.uuid4())
        try:
            r = httpx.request(method, f"{self.base}{path}", json=json_body, headers=headers, timeout=120)
        except httpx.ConnectError:
            con.print(f"[red]control plane unreachable at {self.base}[/] (set FORGE_CONTROLPLANE_URL)")
            raise typer.Exit(2) from None
        if r.status_code >= 400:
            con.print(f"[red]{r.status_code}[/] {r.text}")
            raise typer.Exit(1)
        return r.json()


api = API()


def _ms(v) -> str:
    return "—" if v is None else f"{v * 1000:.0f}ms"


# ====================================================================== model
@model_app.command("train")
def model_train(
    config: Optional[Path] = typer.Option(None, help="YAML file with TrainConfig fields"),
    base_model: Optional[str] = typer.Option(None), dataset: Optional[str] = typer.Option(None),
    name: Optional[str] = typer.Option(None, help="Model name (registry)"), qlora: Optional[bool] = typer.Option(None),
    epochs: Optional[float] = typer.Option(None),
    dry_run: bool = typer.Option(False, help="Validate data and print the GPU memory plan only"),
):
    """Fine-tune a base model with LoRA/QLoRA on a chat dataset."""
    from forge.training.train import TrainConfig, train

    cfg_d = yaml.safe_load(config.read_text()) if config else {}
    for k, v in (("base_model", base_model), ("dataset", dataset), ("output_name", name), ("qlora", qlora),
                 ("epochs", epochs)):
        if v is not None:
            cfg_d[k] = v
    valid = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in cfg_d.items() if k in valid})
    m = train(cfg, dry_run=dry_run)
    plan = m["memory_plan"]
    t = Table(title=f"Run {m['run_id']} ({'dry run' if dry_run else m['status']})")
    t.add_column("item"), t.add_column("value", justify="right")
    t.add_row("dataset", f"{m['dataset']['train']} train / {m['dataset']['val']} val "
                         f"({m['dataset']['duplicates_removed']} dupes removed)")
    t.add_row("dataset sha256", m["dataset"]["sha256"][:16] + "…")
    t.add_row("trainable params", f"{plan['trainable_params_m']}M ({plan['trainable_pct']}%)")
    for k in ("weights_gb", "lora_and_optimizer_gb", "activations_gb", "logits_gb", "total_gb"):
        t.add_row(k, f"{plan[k]} GB")
    t.add_row("fits", ", ".join(g for g, ok in plan["fits"].items() if ok) or "none")
    if not dry_run:
        t.add_row("final loss", str(m["metrics"]["final_train_loss"]))
        t.add_row("best val loss", str(m["metrics"]["best_val_loss"]))
        t.add_row("adapter", m["adapter_uri"])
    con.print(t)


@model_app.command("evaluate")
def model_evaluate(
    name: str, version: str,
    dataset: str = typer.Option("data/sample/eval.jsonl"),
    url: Optional[str] = typer.Option(None, help="OpenAI-compatible endpoint; defaults to the registry serving_url"),
    concurrency: int = typer.Option(4), gpu_hourly_usd: float = typer.Option(0.80),
    record: bool = typer.Option(True, help="Write headline metrics to the registry"),
):
    """Measure quality, latency and cost of a model version on an eval set."""
    from forge.evaluation.harness import evaluate

    if not url:
        url = next((m["serving_url"] for m in api("GET", f"/api/models?name={name}") if m["version"] == version), None)
        if not url:
            con.print("[red]no serving_url; pass --url[/]")
            raise typer.Exit(1)
    rep = evaluate(url, name, version, dataset, concurrency=concurrency, gpu_hourly_usd=gpu_hourly_usd)
    path = rep.save(settings().artifacts / "evals")
    h = rep.headline()
    t = Table(title=f"{name}:{version} on {dataset} (n={rep.n}, errors={rep.errors})")
    t.add_column("metric"), t.add_column("value", justify="right")
    for k, v in h.items():
        t.add_row(k, f"{v}")
    con.print(t)
    con.print(f"report: {path}")
    if record:
        api("POST", f"/api/models/{name}/{version}/metrics", h)
        con.print("[green]metrics recorded in registry[/]")


@model_app.command("register")
def model_register(name: str, version: str, base_model: str = typer.Option(...),
                   adapter_uri: Optional[str] = None, serving_url: Optional[str] = None,
                   run: Optional[str] = typer.Option(None, help="Training run id: pulls lineage from its manifest")):
    """Register an immutable model version (with data lineage when --run is given)."""
    body = {"name": name, "version": version, "base_model": base_model, "adapter_uri": adapter_uri,
            "serving_url": serving_url}
    if run:
        man = json.loads((settings().artifacts / "runs" / run / "manifest.json").read_text())
        body.update(run_id=run, dataset_hash=man["dataset"]["sha256"], adapter_uri=adapter_uri or man["adapter_uri"],
                    metrics={"train_loss": man["metrics"]["final_train_loss"],
                             "val_loss": man["metrics"]["best_val_loss"]})
    mv = api("POST", "/api/models", body)
    con.print(f"[green]registered[/] {mv['name']}:{mv['version']} stage={mv['stage']}")


@model_app.command("list")
def model_list(name: Optional[str] = None):
    """List registered model versions."""
    t = Table()
    for c in ("model", "version", "stage", "accuracy", "ttft p95", "p95", "cost/req", "base"):
        t.add_column(c)
    for m in api("GET", "/api/models" + (f"?name={name}" if name else "")):
        mt = m["metrics"]
        t.add_row(m["name"], m["version"], m["stage"], f"{mt['accuracy']:.1%}" if "accuracy" in mt else "—",
                  _ms(mt.get("ttft_p95_s")), _ms(mt.get("latency_p95_s")),
                  f"${mt['cost_per_request']:.5f}" if "cost_per_request" in mt else "—", m["base_model"])
    con.print(t)


@model_app.command("stage")
def model_stage(name: str, version: str, stage: str, reason: str = ""):
    """Move a version through its lifecycle (candidate, staging, production, archived, rejected)."""
    mv = api("POST", f"/api/models/{name}/{version}/stage", {"stage": stage, "reason": reason})
    con.print(f"{name}:{version} → {mv['stage']}")


@model_app.command("compare")
def model_compare(name: str, base: str, candidate: str):
    """Is the candidate actually better? Quality vs latency vs cost, with a ship decision."""
    res = api("GET", f"/api/models/{name}/compare?base={base}&candidate={candidate}")
    t = Table(title=f"{name}: {base} vs {candidate}")
    for c in ("metric", base, candidate, "Δ"):
        t.add_column(c, justify="right" if c != "metric" else "left")
    for r in res["rows"]:
        arrow = {True: "[green]↑ better[/]", False: "[red]↓ worse[/]", None: "="}[r["better"]]
        t.add_row(r["label"], r["base"], r["candidate"], f"{r['delta_pct']:+.1f}% {arrow}")
    con.print(t)
    color = {"SHIP_CANARY": "green", "NEEDS_REVIEW": "yellow", "BLOCK": "red"}[res["decision"]]
    con.print(f"[bold {color}]{res['decision']}[/]: " + "; ".join(res["reasons"]))


@model_app.command("deploy")
def model_deploy(deployment: str, version: str, weight: int = typer.Option(10, help="Canary traffic %"),
                 model: Optional[str] = typer.Option(None, help="Create the deployment with this model if new")):
    """Canary a version on a deployment (or create the deployment with it as stable)."""
    existing = {d["name"] for d in api("GET", "/api/deployments")}
    if deployment not in existing:
        d = api("POST", "/api/deployments", {"name": deployment, "model": model or deployment, "version": version})
        con.print(f"[green]created[/] {d['name']} stable={d['stable_version']}")
        return
    d = api("POST", f"/api/deployments/{deployment}/canary", {"version": version, "weight": weight}, idem=True)
    con.print(f"[green]canary[/] {version} at {d['canary_weight']}% on {deployment} (rev {d['revision']})")


@model_app.command("weight")
def model_weight(deployment: str, weight: int):
    """Shift canary traffic."""
    d = api("POST", f"/api/deployments/{deployment}/weight", {"weight": weight}, idem=True)
    con.print(f"canary {d['canary_version']} now {d['canary_weight']}% (rev {d['revision']})")


@model_app.command("promote")
def model_promote(deployment: str):
    """Promote the canary to stable."""
    d = api("POST", f"/api/deployments/{deployment}/promote", {}, idem=True)
    con.print(f"[green]promoted[/] {d['stable_version']} on {deployment} (rev {d['revision']})")


@model_app.command("rollback")
def model_rollback(deployment: str, reason: str = typer.Option("manual rollback"),
                   to: Optional[str] = typer.Option(None, help="Explicit target version")):
    """Abort the canary, or revert stable to the previous production version."""
    d = api("POST", f"/api/deployments/{deployment}/rollback", {"reason": reason, "version": to}, idem=True)
    con.print(f"[yellow]rolled back[/] {deployment}: stable={d['stable_version']} canary={d['canary_version']} "
              f"(rev {d['revision']})")


# ====================================================================== agent / incidents
@agent_app.command("run")
def agent_run(question: str, deployment: str = "qwen-farm"):
    """Ask the AI SRE a question, e.g. "Why is production slow?"."""
    con.print(api("POST", "/api/agent/ask", {"question": question, "deployment": deployment})["answer"])


def _show_incident(i: dict) -> None:
    con.rule(f"{i['id']}  [{i['state']}]")
    con.print(f"[bold]{i['title']}[/] ({i['severity']}, {i['source']})")
    if d := i.get("diagnosis"):
        con.print(f"\n[bold]{d['summary']}[/]\n{d['root_cause']}")
        for e in d["evidence"]:
            con.print(f"  • {e}")
        con.print(f"  confidence {d['confidence']:.0%} · investigator {d['investigator']} · "
                  f"{len(d.get('tool_calls', []))} tool calls")
    if a := i.get("proposed_action"):
        con.print(f"\n[bold]Proposed:[/] {a['action']} {json.dumps(a['args'])}\n  {a['rationale']}")
    con.print()
    for t in i.get("timeline", []):
        con.print(f"  {time.strftime('%H:%M:%S', time.localtime(t['at']))}  {t.get('phase') or '':<18} "
                  f"{t.get('note') or ''}")


def _follow(iid: str, until: tuple[str, ...], timeout: float = 300) -> dict:
    deadline, seen = time.time() + timeout, ""
    while time.time() < deadline:
        i = api("GET", f"/api/incidents/{iid}")
        if i["state"] != seen:
            con.print(f"  → {i['state']}")
            seen = i["state"]
        if i["state"] in until:
            return i
        time.sleep(2)
    return api("GET", f"/api/incidents/{iid}")


@incident_app.command("investigate")
def incident_investigate(deployment: str = typer.Argument("qwen-farm"), title: str = "Manual investigation"):
    """Open an incident and let the agent investigate it."""
    i = api("POST", "/api/incidents", {"deployment": deployment, "title": title})
    con.print(f"opened {i['id']}")
    _show_incident(_follow(i["id"], ("AWAITING_APPROVAL", "ESCALATED", "RESOLVED")))


@incident_app.command("list")
def incident_list():
    t = Table()
    for c in ("id", "state", "severity", "title", "opened"):
        t.add_column(c)
    for i in api("GET", "/api/incidents"):
        t.add_row(i["id"], i["state"], i["severity"], i["title"], time.strftime("%H:%M:%S",
                                                                                 time.localtime(i["created_at"])))
    con.print(t)


@incident_app.command("show")
def incident_show(iid: str):
    _show_incident(api("GET", f"/api/incidents/{iid}"))


@incident_app.command("approve")
def incident_approve(iid: str, comment: str = "", follow: bool = True):
    """Approve the agent's proposed action (human-in-the-loop)."""
    api("POST", f"/api/incidents/{iid}/decision", {"approved": True, "comment": comment})
    con.print("[green]approved[/]; executing and verifying…")
    if follow:
        _show_incident(_follow(iid, ("RESOLVED", "VERIFY_FAILED", "ESCALATED")))


@incident_app.command("reject")
def incident_reject(iid: str, comment: str = ""):
    api("POST", f"/api/incidents/{iid}/decision", {"approved": False, "comment": comment})
    con.print("[yellow]rejected[/]")


# ====================================================================== platform
@platform_app.command("status")
def platform_status():
    """One-screen production overview."""
    o = api("GET", "/api/overview")
    m = o["metrics"]
    t = Table(title="Production", show_header=False)
    t.add_column(), t.add_column(justify="right")
    rows = [("Requests/sec", f"{m['rps']:.1f}" if m["rps"] is not None else "—"),
            ("TTFT p95", _ms(m["ttft_p95_s"])), ("P95 latency", _ms(m["latency_p95_s"])),
            ("Tokens/sec", f"{m['tokens_per_s']:.0f}" if m["tokens_per_s"] is not None else "—"),
            ("Error rate", f"{(m['error_rate'] or 0):.1%}"),
            ("GPU utilization", f"{m['gpu_util_pct']:.0f}%" if m["gpu_util_pct"] is not None else "—"),
            ("KV cache", f"{m['kv_cache']:.0%}" if m["kv_cache"] is not None else "—"),
            ("Queue depth", f"{m['queue_depth']:.0f}" if m["queue_depth"] is not None else "—"),
            ("Cost/hour", f"${m['cost_per_hour_usd']:.3f}" if m["cost_per_hour_usd"] is not None else "—")]
    for r in rows:
        t.add_row(*r)
    con.print(t)
    for d in o["deployments"]:
        canary = f", canary {d['canary_version']} {d['canary_weight']}%" if d["canary_version"] else ""
        con.print(f"[bold]{d['name']}[/] {d['status']}: stable {d['stable_version']}{canary} (rev {d['revision']})")
    con.print(f"open incidents: {o['open_incidents']}")


# ====================================================================== demo
SIMS = {"v1": "http://sim-v1:8000", "v2": "http://sim-v2:8000", "v3": "http://sim-v3:8000"}


@demo_app.command("setup")
def demo_setup(base_model: str = "Qwen/Qwen2.5-1.5B-Instruct", sims: str = typer.Option(
        json.dumps(SIMS), help="JSON map version->serving URL")):
    """Register v1-v3, evaluate each, deploy v1, canary+promote v2. Leaves v3 as a candidate."""
    urls = json.loads(sims)
    existing = {m["version"] for m in api("GET", "/api/models?name=qwen-farm")}
    for v, url in urls.items():
        if v not in existing:
            api("POST", "/api/models", {"name": "qwen-farm", "version": v, "base_model": base_model,
                                        "serving_url": url, "adapter_uri": f"artifacts/runs/qwen-farm-{v}/adapter",
                                        "tags": {"source": "demo"}})
        con.rule(f"evaluate {v}")
        model_evaluate("qwen-farm", v, dataset="data/sample/eval.jsonl", url=url, concurrency=4,
                       gpu_hourly_usd=0.80, record=True)
    deployments = {d["name"] for d in api("GET", "/api/deployments")}
    if "qwen-farm" not in deployments:
        api("POST", "/api/deployments", {"name": "qwen-farm", "model": "qwen-farm", "version": "v1"})
        con.rule("v1 → v2")
        model_compare("qwen-farm", "v1", "v2")
        verdict = api("GET", "/api/models/qwen-farm/compare?base=v1&candidate=v2")["decision"]
        if verdict == "BLOCK":
            con.print("[red]ship gate blocked v2; leaving v1 in production[/]")
            raise typer.Exit(1)
        if verdict == "NEEDS_REVIEW":
            con.print("[yellow]Reviewer decision (demo): accept the latency/cost trade for the quality gain.[/]")
        api("POST", "/api/deployments/qwen-farm/canary", {"version": "v2", "weight": 10}, idem=True)
        api("POST", "/api/deployments/qwen-farm/promote", {}, idem=True)
    con.rule("v2 → v3 (candidate)")
    model_compare("qwen-farm", "v2", "v3")
    con.print("\n[bold]Ready.[/] Production = v2. Start load (`make load`), then `forge demo incident`.")


@demo_app.command("incident")
def demo_incident(weight: int = 50, wait_alert_s: int = 150, approve: bool = typer.Option(
        False, help="Auto-approve the agent's recommendation (otherwise approve it yourself)")):
    """Canary v3 → SLO violation → alert → agent diagnosis → approval → rollback → verify."""
    v3 = next(m for m in api("GET", "/api/models?name=qwen-farm") if m["version"] == "v3")
    if v3["stage"] == "rejected":
        api("POST", "/api/models/qwen-farm/v3/stage", {"stage": "candidate", "reason": "demo re-run"})
    con.print(f"[bold]1.[/] Canary v3 at {weight}% (offline eval looked fine at low concurrency)…")
    api("POST", "/api/deployments/qwen-farm/canary", {"version": "v3", "weight": weight}, idem=True)
    con.print(f"[bold]2.[/] Waiting up to {wait_alert_s}s for Prometheus to fire InferenceTTFTSLOViolation…")
    deadline, inc = time.time() + wait_alert_s, None
    while time.time() < deadline and not inc:
        inc = next((i for i in api("GET", "/api/incidents") if i["source"] == "alertmanager"
                    and i["state"] not in ("RESOLVED", "REJECTED", "ESCALATED", "VERIFY_FAILED")), None)
        time.sleep(3)
    if not inc:
        con.print("   no alert yet; opening the incident manually")
        inc = api("POST", "/api/incidents", {"deployment": "qwen-farm", "title": "TTFT SLO violation (manual)",
                                             "severity": "critical"})
    con.print(f"[bold]3.[/] Incident {inc['id']}: agent investigating…")
    i = _follow(inc["id"], ("AWAITING_APPROVAL", "ESCALATED", "RESOLVED"))
    _show_incident(i)
    if i["state"] != "AWAITING_APPROVAL":
        return
    if approve:
        incident_approve(i["id"], comment="demo auto-approve")
    else:
        con.print(f"\n[bold]4.[/] Awaiting your decision: `forge incident approve {i['id']}` (or use the UI)")


@demo_app.command("cost")
def demo_cost(multiplier: float = 1.9, settle_s: int = 300):
    """A 'prompt change' nearly doubles output tokens on production; the agent explains the cost jump."""
    url = os.environ.get("FORGE_SIM_ADMIN", "http://sim-v2:8000")
    httpx.post(f"{url}/admin/faults", json={"output_tokens_multiplier": multiplier}, timeout=5).raise_for_status()
    con.print(f"output tokens x{multiplier} on v2; letting {settle_s}s of traffic accumulate…")
    time.sleep(settle_s)
    agent_run("Why did inference cost increase over the last 5 minutes compared to the 5 minutes before?")
    httpx.post(f"{url}/admin/faults", json={"reset": True}, timeout=5)


# ====================================================================== serve
@serve_app.command("gateway")
def serve_gateway(port: int = 8000):
    import uvicorn

    uvicorn.run("forge.gateway.app:app", host="0.0.0.0", port=port, log_level="info")


@serve_app.command("controlplane")
def serve_controlplane(port: int = 8080):
    import uvicorn

    uvicorn.run("forge.controlplane.app:app", host="0.0.0.0", port=port, log_level="info")


@serve_app.command("simulator")
def serve_simulator(port: int = 8000):
    import uvicorn

    uvicorn.run("forge.serving.simulator:app", host="0.0.0.0", port=port, log_level="warning")


@serve_app.command("mcp")
def serve_mcp(transport: str = typer.Option("stdio", help="stdio | streamable-http")):
    from forge.mcp_server import server

    server.run(transport=transport)


if __name__ == "__main__":
    app()

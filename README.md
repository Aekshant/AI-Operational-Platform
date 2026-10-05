# ForgeAI

A self-hosted platform for **fine-tuning, evaluating, deploying, serving and operating small language models**, with an AI SRE agent that diagnoses production incidents and fixes them once a human approves.

```
train → evaluate → register → deploy (canary) → serve → observe → diagnose → approve → rollback → verify
```

The agent is not the product. It is the platform's operator. Every action it takes goes through the same
control-plane API, RBAC, audit log and idempotency guarantees as a human operator.

---

## What it does

```text
                         ┌──────────────────────────────┐
                         │  Web UI · forge CLI · MCP    │
                         └──────────────┬───────────────┘
                                        ▼
┌─────────────────────────────── control plane ──────────────────────────────────┐
│  Training (LoRA/QLoRA) → Evaluation (quality/latency/cost) → Ship gate         │
│  Model registry (lineage, stages) → Deploy controller (canary, revisions)      │
│  Incidents + AI SRE (LangGraph state machine, durable, human approval)         │
└───────────────────────────────┬────────────────────────────────────────────────┘
                                │ desired routing (polled, last-known-good)
                                ▼
┌──────────────────────────── inference gateway ─────────────────────────────────┐
│ auth/RBAC → tenant rate limit → guardrails → weighted canary routing           │
│ → circuit breaker → bulkhead + priority queue (load shedding)                  │
│ → upstream with TTFT timeout, retry-elsewhere, fallback → streaming relay      │
└───────────────┬─────────────────────────┬──────────────────────────────────────┘
                ▼                         ▼
        model server v2 (stable)   model server v3 (canary)     vLLM, or the bundled simulator
                │                         │
                └──── /metrics ───────────┴──► Prometheus ─► Alertmanager ─► webhook ─► AI SRE
      OTLP traces + logs ─► OTel Collector ─► Tempo / Loki ─► Grafana
```

### The incident loop (runs end to end locally)

1. v3 passes offline evaluation: +2.5 pts accuracy, latency within budget, ship gate says `SHIP_CANARY`.
2. It goes out as a 50% canary. v3 needs ~30% more GPU memory, so only 4 sequences fit in its batch. Under production concurrency it queues, and TTFT p95 goes from 195 ms to several seconds.
3. Prometheus fires `InferenceTTFTSLOViolation` → Alertmanager → control-plane webhook → incident opened (de-duplicated by fingerprint).
4. The agent investigates with read-only tools: SLOs → *what changed?* → per-version TTFT, GPU memory, KV cache, queue depth → diagnosis with cited evidence.
5. It recommends `rollback_model` and the graph **suspends** at the approval gate (`AWAITING_APPROVAL`).
6. A human approves (CLI or UI). The rollback executes with an idempotency key, v3 is marked `rejected` in the registry, and the agent verifies the SLOs are back for 3 consecutive checks → `RESOLVED`.

Output from a run of the local stack (`make incident`, abridged):

```
qwen-farm: v3 regressed TTFT vs v2; roll back v3.
Model version v3 (introduced by 'start_canary') is the regression source: TTFT on v3 is significantly worse
than on v2 under the same traffic, alongside +39% GPU memory. Consistent with the new version needing more
memory, so fewer sequences fit in the batch and requests queue before prefill (11 waiting on v3 vs 0 on v2).
  • SLO violations on qwen-farm: ttft_p95, latency_p95 (TTFT p95 2.7095s vs 0.5s, error rate 0.0)
  • Change 0.5 min ago: start_canary {"version": "v3", "weight": 50, "stable": "v2"} (revision 6)
  • TTFT p95: v3 4.7963s vs v2 0.8037s (+497%)
  • GPU memory: v3 19.56 GB vs v2 14.06 GB (+39%); KV cache v3 18%
  • Queue (vLLM waiting): v3 11 vs v2 0
  confidence 85% · investigator heuristic · 7 tool calls
Proposed: rollback_model {"deployment": "qwen-farm", "reason": "v3 TTFT regression vs v2"}
  10:26:28  AWAITING_APPROVAL  Recommends rollback_model
  10:26:31  EXECUTING          Executing rollback_model (approved by platform:admin)
  10:26:31  VERIFYING          rollback_model applied (revision 7); verifying SLOs
  10:28:32  RESOLVED           Recovery confirmed: TTFT p95 0.1951s, error rate 0.0 within SLO for 3 consecutive checks
```

### "Is v2 actually better than v1?"

```
$ forge model compare qwen-farm v1 v2
 Accuracy          80.0%     87.5%   +9.4% ↑ better
 TTFT p95          143ms     174ms  +21.7% ↓ worse
 P95 latency      1127ms    1514ms  +34.4% ↓ worse
 Cost/request   $0.00005  $0.00007  +32.0% ↓ worse
NEEDS_REVIEW: quality improved +7.5 pts, but p95 latency increased 34% (budget 20%); cost/request increased 32% (budget 25%)
```

The ship gate ([forge/evaluation/compare.py](forge/evaluation/compare.py)) turns that into an explicit policy:
`BLOCK` for quality regressions, `NEEDS_REVIEW` when a quality gain costs more latency or money than the
budget allows, `SHIP_CANARY` otherwise.

### "Why did inference cost go up?"

Cost = requests × tokens/request × price/token. The agent splits the change log-additively, so each
factor's share is exact, and names the driver:
*"Cost +58.7% vs baseline. Traffic -1.5%, tokens/request +63.3% (completion 134 → 227), unit price -1.3%.
Primary driver: tokens per request (106% of the change)."* (`make cost`, which raises output length ×1.9 on production)
See [forge/agent/cost.py](forge/agent/cost.py).

---

## Quickstart (no GPU needed)

Requirements: Docker, [uv](https://docs.astral.sh/uv/).

```bash
make install            # Python 3.11 venv + deps
make test               # 35 tests, including real gateway + simulator processes
make up                 # build and start the platform + observability stack
make demo               # register + evaluate v1..v3, deploy v1, ship v2 through the gate
make load               # background traffic (open-loop Poisson, 6 rps, 3 tenants)
make incident           # canary v3 → alert → agent diagnosis → waits for you
make approve ID=inc-…   # human-in-the-loop approval → rollback → verified recovery
make cost               # output tokens ×1.9 → ask the agent why cost went up
```

| URL | |
|---|---|
| http://localhost:8080 | ForgeAI UI: production tiles, versions, rollouts, incidents (approve/reject), ask the AI SRE |
| http://localhost:3000 | Grafana: *ForgeAI Inference Platform* dashboard (traces in Tempo, logs in Loki) |
| http://localhost:8000 | Gateway: OpenAI-compatible `/v1/chat/completions` (`Authorization: Bearer dev-admin`) |
| http://localhost:9090 | Prometheus (alerts tab shows the SLO rules) |

If those ports are taken, copy `deploy/.env.example` to `deploy/.env` and change them.

**Let Claude drive the investigation**: set `ANTHROPIC_API_KEY` in `deploy/.env`. With no key, the
deterministic runbook investigator is used. It is also the fallback if the LLM call fails or declines.

```bash
curl localhost:8000/v1/chat/completions -H 'Authorization: Bearer dev-acme' -H 'Content-Type: application/json' \
  -d '{"model":"qwen-farm","stream":true,"messages":[{"role":"user","content":"What does NPK stand for?"}]}'
```

---

## The CLI

```bash
forge model train --config data/sample/train_config.yaml [--dry-run]   # LoRA/QLoRA SFT; dry-run = GPU memory plan
forge model register qwen-farm v4 --base-model Qwen/Qwen2.5-1.5B-Instruct --run <run_id>
forge model evaluate qwen-farm v4                    # quality, TTFT, p95, tokens/s, $/request → registry
forge model compare  qwen-farm v3 v4                 # ship gate
forge model deploy   qwen-farm v4 --weight 10        # canary
forge model weight   qwen-farm 50
forge model promote  qwen-farm
forge model rollback qwen-farm --reason "..."
forge model stage    qwen-farm v3 candidate          # re-qualify a rejected version

forge agent run "Why is production slow?"
forge incident investigate qwen-farm
forge incident approve <id> | reject <id>
forge platform status
forge serve mcp                                      # expose the toolbox to Claude Code / Desktop
```

`claude mcp add forgeai -- forge serve mcp` gives Claude Code the same read tools as the in-house agent.
Writes are only available as `request_action`, which files an incident into the same approval gate.

---

## Where each production pattern lives

| Pattern | Implementation |
|---|---|
| LoRA / QLoRA SFT, assistant-only loss masking, grad accumulation, checkpoints, val loss | [forge/training/train.py](forge/training/train.py) |
| GPU memory planning (weights, LoRA+Adam, activations, logits) | `memory_plan()` in the same file; matches PEFT's param count exactly |
| Dataset validation, dedupe, deterministic split, content hash → lineage | [forge/training/dataset.py](forge/training/dataset.py) |
| Evaluation: quality + TTFT + latency + throughput + $/token from GPU-hour cost | [forge/evaluation/harness.py](forge/evaluation/harness.py) |
| Ship gate with explicit budgets | [forge/evaluation/compare.py](forge/evaluation/compare.py) |
| Model registry: immutable versions, guarded stage transitions, one production version, MLflow mirror | [forge/registry/registry.py](forge/registry/registry.py) |
| Canary / promote / rollback, optimistic concurrency (revisions), idempotency keys, event log | [forge/deploy/controller.py](forge/deploy/controller.py) |
| Eventual consistency + graceful degradation (last-known-good routing, staleness metric) | [forge/gateway/routing.py](forge/gateway/routing.py) |
| Token-bucket rate limiting per tenant | [forge/gateway/ratelimit.py](forge/gateway/ratelimit.py) |
| Bulkhead, bounded queue, priority-aware load shedding, backpressure (429 + Retry-After) | [forge/gateway/admission.py](forge/gateway/admission.py) |
| Circuit breaker (rolling window, half-open probes) | [forge/gateway/breaker.py](forge/gateway/breaker.py) |
| Timeouts (TTFT), retry with full-jitter backoff to a *different* backend, fallback | [forge/gateway/app.py](forge/gateway/app.py) |
| Retry only before the first token; mid-stream failures reported in-band | same |
| Streaming relay with TTFT / TPOT / tokens / cost attribution per version and tenant | same |
| AuthN, RBAC (viewer/operator/admin), guardrails, audit log | [forge/gateway/auth.py](forge/gateway/auth.py), [guardrails.py](forge/gateway/guardrails.py), [forge/db.py](forge/db.py) |
| LLM-serving realism: continuous batching, `max_num_seqs`, KV-cache pressure, DCGM metrics, fault injection | [forge/serving/simulator.py](forge/serving/simulator.py) |
| SLO recording rules + alerts, Alertmanager → agent webhook | [deploy/prometheus/](deploy/prometheus/) |
| Distributed tracing (admission / prefill spans), OTLP logs, Grafana dashboard as code | [forge/telemetry.py](forge/telemetry.py), [scripts/gen_dashboard.py](scripts/gen_dashboard.py) |
| Durable agent state machine, checkpoints, `interrupt()` HITL, resume after restart | [forge/agent/graph.py](forge/agent/graph.py) |
| Tool authorization (read vs write), allowlisted actions, prompt-injection containment | [forge/agent/tools.py](forge/agent/tools.py), [investigator.py](forge/agent/investigator.py) |
| MCP server | [forge/mcp_server.py](forge/mcp_server.py) |
| Kubernetes: vLLM + LoRA per version, GPU scheduling, KEDA on queued sequences, PDB, HPA, NetworkPolicy, ServiceMonitor/PodMonitor, PrometheusRule | [deploy/helm/forgeai/](deploy/helm/forgeai/) |

### Agent safety model

Prompt injection is treated as inevitable, not preventable. Tenants control prompts, prompts reach logs,
and logs reach the agent. So:

- Investigation uses **read-only** tools. Tool output is wrapped as untrusted data.
- The LLM's only output is `submit_diagnosis`, whose `action` is an **enum** of allowlisted write tools.
- The graph re-validates the action, strips unknown arguments, and **refuses actions targeting a different deployment** than the incident.
- Nothing executes without a human decision recorded in the checkpoint and the audit log.
- Writes carry `incident_id:action` idempotency keys, so a retried step cannot roll back twice.

The worst a successful injection can produce is a wrong *recommendation*, shown to a human with its evidence.

---

## What is simulated, and how to make it real

Everything except the GPU is real. The **model-server simulator** speaks the same OpenAI API and emits the
same `vllm:*` and `DCGM_*` metric names as vLLM with the DCGM exporter. Its latency comes from a capacity
model (batch slots, KV pressure, prefill cost), not random sleeps. That is why the canary incident above
emerges from load instead of being scripted. Model *quality* in the simulator is a configured accuracy
against the eval answer key.

On a GPU machine:

```bash
uv pip install -e ".[train,qlora]"
forge model train --config data/sample/train_config.yaml          # → .forge/artifacts/runs/<run>/adapter
vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-lora \
     --lora-modules qwen-farm-v4=.forge/artifacts/runs/<run>/adapter --max-num-seqs 64
forge model register qwen-farm v4 --base-model Qwen/Qwen2.5-1.5B-Instruct --run <run> \
     --serving-url http://<vllm-host>:8000
forge model evaluate qwen-farm v4 && forge model compare qwen-farm v2 v4
```

The training loop has been run end to end on CPU with `HuggingFaceTB/SmolLM2-135M-Instruct`. It saves the
adapter, and loss masking was verified to cover only assistant tokens.

On Kubernetes: `helm install forgeai deploy/helm/forgeai` (needs the NVIDIA GPU Operator, kube-prometheus-stack and KEDA).

---

## Known limits / next steps

- **State store** is SQLite (single control-plane writer). Postgres is the next step before running more than one replica.
- **Rate limits** are per gateway replica. Redis-backed buckets are needed for exact global limits.
- **Deploy controller** manages routing, not pods. In Kubernetes the next step is reconciling `modelServers` from the registry (an operator, or Argo CD from a generated values file, GitOps).
- **Eval** uses reference-match scoring. LLM-as-judge and pairwise preference evals fit behind the same `SCORERS` interface.
- **Agent memory** across incidents (past diagnoses as context) is not implemented yet.

## Layout

```
forge/
  training/      dataset prep, LoRA/QLoRA SFT, memory planner
  evaluation/    harness, scorers, ship gate
  registry/      model versions + lifecycle
  deploy/        rollout controller
  gateway/       inference gateway + reliability primitives
  serving/       vLLM-compatible simulator
  agent/         tools, investigators, cost attribution, LangGraph incident graph
  controlplane/  API + web UI
  mcp_server.py  MCP server
  cli.py         `forge`
deploy/          docker-compose, Prometheus/Alertmanager/OTel/Tempo/Grafana, Helm chart
data/sample/     agronomy SFT + eval sets ("qwen-farm")
scripts/         load generator, dashboard generator
tests/           unit + process-level integration tests
```

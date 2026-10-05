"""A vLLM-compatible model server simulator for running the platform without a GPU.

It is not a toy echo server. It models the things that make LLM serving operationally
interesting, so the gateway, dashboards, alerts and agent face realistic behavior:

  * continuous batching with a hard `max_num_seqs` concurrency limit; excess requests wait
    (-> `vllm:num_requests_waiting`, i.e. queue depth)
  * prefill cost proportional to prompt length; TTFT inflates with KV-cache pressure
  * decode time per token (TPOT) that degrades as the batch grows
  * KV cache utilisation derived from tokens resident in running sequences
  * DCGM-style GPU utilisation and framebuffer memory
  * per-version answer quality against an eval answer key (deterministic, reproducible)
  * fault injection (`POST /admin/faults`) to create incidents on demand

It exposes the same OpenAI API and the same metric names as vLLM, so swapping in a real
`vllm serve` changes nothing upstream. Configure with SIM_* env vars.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Profile:
    model: str = os.environ.get("SIM_MODEL", "qwen-farm")
    version: str = os.environ.get("SIM_VERSION", "v1")
    prefill_ms_per_1k: float = _f("SIM_PREFILL_MS_PER_1K", 60)  # prompt processing cost
    ttft_base_ms: float = _f("SIM_TTFT_BASE_MS", 90)
    tpot_ms: float = _f("SIM_TPOT_MS", 9)  # decode time per output token at batch size 1
    batch_penalty: float = _f("SIM_BATCH_PENALTY", 0.015)  # TPOT growth per extra running seq
    max_num_seqs: int = int(_f("SIM_MAX_NUM_SEQS", 16))
    kv_capacity_tokens: int = int(_f("SIM_KV_CAPACITY_TOKENS", 60000))
    mean_output_tokens: float = _f("SIM_MEAN_OUTPUT_TOKENS", 120)
    accuracy: float = _f("SIM_ACCURACY", 0.8)
    error_rate: float = _f("SIM_ERROR_RATE", 0.0)
    gpu_mem_base_gb: float = _f("SIM_GPU_MEM_BASE_GB", 14)
    gpu_mem_total_gb: float = _f("SIM_GPU_MEM_TOTAL_GB", 24)


@dataclass
class Faults:
    ttft_multiplier: float = 1.0
    tpot_multiplier: float = 1.0
    output_tokens_multiplier: float = 1.0
    error_rate: float = 0.0
    extra_gpu_mem_gb: float = 0.0
    capacity_multiplier: float = 1.0  # <1 shrinks effective max_num_seqs (e.g. KV cache fragmentation)


P = Profile()
F = Faults()
_initial = os.environ.get("SIM_FAULTS")
if _initial:
    F = Faults(**json.loads(_initial))

ANSWERS: dict[str, str] = {}
_key = os.environ.get("SIM_ANSWER_KEY", "data/sample/eval.jsonl")
if Path(_key).exists():
    for line in Path(_key).read_text().splitlines():
        if line.strip():
            ex = json.loads(line)
            ANSWERS[ex["prompt"].strip()] = ex["reference"]
# Rank questions by a version-independent hash: a version with accuracy a answers exactly the
# easiest round(a * N) correctly. Better versions are supersets of worse ones, as with real fine-tunes
# trained on more/better data, and offline accuracy is exact rather than sampling noise.
_RANK = {q: i for i, q in enumerate(sorted(ANSWERS, key=lambda q: hashlib.sha256(q.encode()).hexdigest()))}

reg = CollectorRegistry()
L = ["model_name"]
RUNNING = Gauge("vllm:num_requests_running", "Requests in the running batch", L, registry=reg)
WAITING = Gauge("vllm:num_requests_waiting", "Requests waiting for a batch slot", L, registry=reg)
KV = Gauge("vllm:gpu_cache_usage_perc", "KV cache usage (0-1)", L, registry=reg)
TTFT = Histogram("vllm:time_to_first_token_seconds", "TTFT", L, registry=reg,
                 buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10))
TPOT = Histogram("vllm:time_per_output_token_seconds", "TPOT", L, registry=reg,
                 buckets=(0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2))
E2E = Histogram("vllm:e2e_request_latency_seconds", "End to end latency", L, registry=reg,
                buckets=(0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 13, 20, 30))
PROMPT_TOK = Counter("vllm:prompt_tokens", "Prompt tokens processed", L, registry=reg)
GEN_TOK = Counter("vllm:generation_tokens", "Generation tokens produced", L, registry=reg)
SUCCESS = Counter("vllm:request_success", "Finished requests", L + ["finished_reason"], registry=reg)
GPU_UTIL = Gauge("DCGM_FI_DEV_GPU_UTIL", "GPU utilisation %", ["gpu", "modelName"], registry=reg)
GPU_FB = Gauge("DCGM_FI_DEV_FB_USED", "GPU framebuffer used (MiB)", ["gpu", "modelName"], registry=reg)
GPU_FB_FREE = Gauge("DCGM_FI_DEV_FB_FREE", "GPU framebuffer free (MiB)", ["gpu", "modelName"], registry=reg)

app = FastAPI(title=f"forge-sim {P.model}:{P.version}")
served_name = f"{P.model}-{P.version}"


class Engine:
    def __init__(self) -> None:
        self.running = 0
        self.waiting = 0
        self.resident_tokens = 0
        self.cv = asyncio.Condition()

    def capacity(self) -> int:
        return max(1, int(P.max_num_seqs * F.capacity_multiplier))

    async def acquire(self) -> None:
        async with self.cv:
            self.waiting += 1
            try:
                await self.cv.wait_for(lambda: self.running < self.capacity())
            finally:
                self.waiting -= 1
            self.running += 1

    async def release(self) -> None:
        async with self.cv:
            self.running -= 1
            self.cv.notify_all()

    def kv_usage(self) -> float:
        return min(1.0, self.resident_tokens / P.kv_capacity_tokens)


engine = Engine()


def _tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def _respond(prompt: str, max_tokens: int) -> list[str]:
    """Deterministic per (version, prompt): correct answer with probability = accuracy."""
    key = prompt.strip()
    h = int(hashlib.sha256(f"{P.version}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    target = int(P.mean_output_tokens * F.output_tokens_multiplier * (0.6 + 0.8 * h))
    if key in ANSWERS:
        correct = _RANK[key] < round(P.accuracy * len(ANSWERS))
        head = ANSWERS[key] if correct else "I am not certain; it depends on local conditions."
    else:
        head = "Based on the information provided,"
    filler = ("the recommended approach balances soil moisture, crop stage and local weather "
              "so that inputs are applied when the plant can actually use them").split()
    words = head.split()
    i = 0
    while _tokens(" ".join(words)) < target:
        words.append(filler[i % len(filler)])
        i += 1
    words = words[: max(1, min(len(words), int(max_tokens / 1.3)))]
    return [w + " " for w in words]


def _prompt_text(body: dict) -> str:
    if "messages" in body:
        return "\n".join(str(m.get("content", "")) for m in body["messages"])
    return str(body.get("prompt", ""))


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    stream = bool(body.get("stream"))
    prompt = _prompt_text(body)
    max_tokens = int(body.get("max_tokens") or 512)
    prompt_tokens = _tokens(prompt)
    rid = f"cmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    t0 = time.monotonic()

    if random.random() < max(P.error_rate, F.error_rate):
        await asyncio.sleep(0.05)
        raise HTTPException(500, "simulated engine error: CUDA error: an illegal memory access was encountered")

    await engine.acquire()
    engine.resident_tokens += prompt_tokens
    words = _respond(prompt, max_tokens)

    async def generate():
        produced = 0
        reason = "stop"
        try:
            prefill = (P.ttft_base_ms + P.prefill_ms_per_1k * prompt_tokens / 1000) / 1000
            prefill *= F.ttft_multiplier * (1 + 1.5 * engine.kv_usage() ** 2)
            await asyncio.sleep(prefill)
            ttft = time.monotonic() - t0
            TTFT.labels(served_name).observe(ttft)
            last = time.monotonic()
            for i, w in enumerate(words):
                if i:
                    tpot = P.tpot_ms / 1000 * F.tpot_multiplier * (1 + P.batch_penalty * (engine.running - 1))
                    tpot *= 1 + engine.kv_usage()
                    await asyncio.sleep(tpot)
                    now = time.monotonic()
                    TPOT.labels(served_name).observe(now - last)
                    last = now
                t = int(1.3 * (i + 1)) - int(1.3 * i)  # ~1.3 tokens per word
                produced += t
                engine.resident_tokens += t
                yield w, None
            if produced >= max_tokens:
                reason = "length"
            yield None, {"prompt_tokens": prompt_tokens, "completion_tokens": produced,
                         "total_tokens": prompt_tokens + produced, "finish_reason": reason}
        finally:
            engine.resident_tokens -= prompt_tokens + produced
            elapsed = time.monotonic() - t0
            E2E.labels(served_name).observe(elapsed)
            PROMPT_TOK.labels(served_name).inc(prompt_tokens)
            GEN_TOK.labels(served_name).inc(produced)
            SUCCESS.labels(served_name, reason).inc()
            await engine.release()

    is_chat = request.url.path.endswith("chat/completions")
    if not stream:
        text, usage = "", {}
        async for piece, u in generate():
            if piece:
                text += piece
            if u:
                usage = u
        reason = usage.pop("finish_reason")
        choice = ({"index": 0, "message": {"role": "assistant", "content": text.strip()}, "finish_reason": reason}
                  if is_chat else {"index": 0, "text": text.strip(), "finish_reason": reason})
        return JSONResponse({"id": rid, "object": "chat.completion" if is_chat else "text_completion",
                             "created": created, "model": served_name, "choices": [choice], "usage": usage})

    async def sse():
        async for piece, u in generate():
            if piece is not None:
                delta = {"index": 0, "delta": {"content": piece}, "finish_reason": None} if is_chat else \
                        {"index": 0, "text": piece, "finish_reason": None}
                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': served_name, 'choices': [delta]})}\n\n"
            else:
                reason = u.pop("finish_reason")
                end = {"index": 0, "delta": {}, "finish_reason": reason}
                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': served_name, 'choices': [end], 'usage': u})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "forgeai"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "model": served_name}


@app.get("/metrics")
async def metrics():
    RUNNING.labels(served_name).set(engine.running)
    WAITING.labels(served_name).set(engine.waiting)
    KV.labels(served_name).set(engine.kv_usage())
    util = min(100.0, 90 * engine.running / engine.capacity() + (8 if engine.running else 2))
    mem = P.gpu_mem_base_gb + F.extra_gpu_mem_gb + 6 * engine.kv_usage()
    mem = min(mem, P.gpu_mem_total_gb)
    GPU_UTIL.labels("0", "NVIDIA L4 (simulated)").set(round(util, 1))
    GPU_FB.labels("0", "NVIDIA L4 (simulated)").set(mem * 1024)
    GPU_FB_FREE.labels("0", "NVIDIA L4 (simulated)").set((P.gpu_mem_total_gb - mem) * 1024)
    return PlainTextResponse(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/faults")
async def get_faults():
    return {"profile": asdict(P), "faults": asdict(F)}


@app.post("/admin/faults")
async def set_faults(request: Request):
    global F
    body = await request.json()
    F = Faults() if body.get("reset") else Faults(**{**asdict(F), **body})
    return asdict(F)

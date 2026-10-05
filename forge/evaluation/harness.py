"""Evaluation harness: quality, latency and cost for a model version on one dataset.

Talks to any OpenAI-compatible endpoint (vLLM, SGLang, the simulator, or the gateway),
always streaming so TTFT is measured the same way users experience it.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from forge.evaluation.scorers import SCORERS


def load_dataset(path: str | Path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for i, r in enumerate(rows):
        if "prompt" not in r or "reference" not in r:
            raise ValueError(f"{path}:{i + 1}: eval rows need 'prompt' and 'reference'")
    return rows


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class Sample:
    prompt: str
    reference: str
    prediction: str = ""
    ttft_s: float = 0.0
    latency_s: float = 0.0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    error: str | None = None
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    model: str
    version: str
    dataset: str
    n: int
    errors: int
    quality: dict[str, float]
    latency: dict[str, float]
    throughput: dict[str, float]
    cost: dict[str, float]
    created_at: float
    samples: list[dict] = field(default_factory=list)

    def headline(self) -> dict[str, float]:
        """Flat metrics stored in the registry and compared between versions."""
        return {
            "accuracy": round(self.quality["primary"], 4),
            "token_f1": round(self.quality.get("token_f1", 0.0), 4),
            "ttft_p95_s": round(self.latency["ttft_p95"], 4),
            "latency_p50_s": round(self.latency["p50"], 4),
            "latency_p95_s": round(self.latency["p95"], 4),
            "tokens_per_s": round(self.throughput["decode_tokens_per_s"], 2),
            "avg_output_tokens": round(self.throughput["avg_output_tokens"], 1),
            "cost_per_1k_tokens": round(self.cost["per_1k_tokens"], 6),
            "cost_per_request": round(self.cost["per_request"], 6),
            "error_rate": round(self.errors / max(1, self.n), 4),
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.model}-{self.version}-{int(self.created_at)}.json"
        p.write_text(json.dumps(asdict(self), indent=2))
        return p


async def _one(client: httpx.AsyncClient, url: str, served_model: str, s: Sample, headers: dict,
               max_tokens: int) -> None:
    body = {"model": served_model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": s.prompt}]}
    t0 = time.monotonic()
    parts: list[str] = []
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                s.error = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                return
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                for c in chunk.get("choices", []):
                    piece = c.get("delta", {}).get("content")
                    if piece:
                        if not parts:
                            s.ttft_s = time.monotonic() - t0
                        parts.append(piece)
                if u := chunk.get("usage"):
                    s.completion_tokens = u.get("completion_tokens", 0)
                    s.prompt_tokens = u.get("prompt_tokens", 0)
    except httpx.HTTPError as e:
        s.error = f"{type(e).__name__}: {e}"
        return
    s.latency_s = time.monotonic() - t0
    s.prediction = "".join(parts).strip()
    s.completion_tokens = s.completion_tokens or len(parts)


async def run_eval(url: str, model: str, version: str, dataset: str | Path, *, served_model: str | None = None,
                   api_key: str | None = None, concurrency: int = 8, max_tokens: int = 256,
                   primary_scorer: str = "contains", gpu_hourly_usd: float = 0.80) -> EvalReport:
    rows = load_dataset(dataset)
    samples = [Sample(r["prompt"], r["reference"]) for r in rows]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    sem = asyncio.Semaphore(concurrency)
    served = served_model or f"{model}-{version}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5)) as client:
        async def guarded(s: Sample):
            async with sem:
                await _one(client, url.rstrip("/"), served, s, headers, max_tokens)

        t0 = time.monotonic()
        await asyncio.gather(*(guarded(s) for s in samples))
        wall = time.monotonic() - t0

    ok = [s for s in samples if not s.error]
    for s in ok:
        s.scores = {name: fn(s.prediction, s.reference) for name, fn in SCORERS.items()}
    quality = {name: statistics.fmean([s.scores[name] for s in ok]) if ok else 0.0 for name in SCORERS}
    quality["primary"] = quality[primary_scorer]
    quality["primary_scorer"] = primary_scorer  # type: ignore[assignment]

    lat = [s.latency_s for s in ok]
    ttft = [s.ttft_s for s in ok]
    decode = [(s.completion_tokens - 1) / (s.latency_s - s.ttft_s)
              for s in ok if s.completion_tokens > 1 and s.latency_s > s.ttft_s]
    total_out = sum(s.completion_tokens for s in ok)
    total_tok = total_out + sum(s.prompt_tokens for s in ok)
    agg_tps = total_tok / wall if wall else 0.0
    # Cost model: a GPU costs $X/hour regardless; cost/token = GPU-seconds / tokens actually produced.
    per_1k = (gpu_hourly_usd / 3600) / agg_tps * 1000 if agg_tps else 0.0
    avg_tokens = total_tok / len(ok) if ok else 0.0

    return EvalReport(
        model=model, version=version, dataset=str(dataset), n=len(samples), errors=len(samples) - len(ok),
        quality=quality,
        latency={"p50": pct(lat, .5), "p95": pct(lat, .95), "p99": pct(lat, .99),
                 "ttft_p50": pct(ttft, .5), "ttft_p95": pct(ttft, .95)},
        throughput={"decode_tokens_per_s": statistics.fmean(decode) if decode else 0.0,
                    "aggregate_tokens_per_s": agg_tps,
                    "avg_output_tokens": total_out / len(ok) if ok else 0.0, "concurrency": concurrency},
        cost={"gpu_hourly_usd": gpu_hourly_usd, "per_1k_tokens": per_1k, "per_request": per_1k * avg_tokens / 1000},
        created_at=time.time(),
        samples=[asdict(s) for s in samples],
    )


def evaluate(*args, **kwargs) -> EvalReport:
    return asyncio.run(run_eval(*args, **kwargs))

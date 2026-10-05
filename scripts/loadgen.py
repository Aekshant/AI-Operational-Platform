"""Open-loop load generator: Poisson arrivals at a target RPS across tenants.

Open-loop matters: a closed loop (N workers waiting on responses) slows down when the system
slows down, which hides the very queueing collapse we want to observe.
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

import httpx

TENANTS = [("dev-admin", 0.5), ("dev-acme", 0.35), ("dev-free", 0.15)]


async def one(client, url, key, prompt, stats, max_tokens):
    t0 = time.monotonic()
    try:
        r = await client.post(f"{url}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                              json={"model": "qwen-farm", "max_tokens": max_tokens,
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=60)
        stats[f"{r.status_code} {r.headers.get('x-forge-version', '-')}/{r.headers.get('x-forge-route', '-')}"] += 1
    except httpx.HTTPError as e:
        stats[type(e).__name__] += 1
    stats["_lat"] += time.monotonic() - t0
    stats["_n"] += 1


async def main(url, rps, duration, max_tokens):
    prompts = [json.loads(l)["prompt"] for l in Path("data/sample/eval.jsonl").read_text().splitlines() if l]
    stats: Counter = Counter()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000)) as client:
        tasks, start, last_print = set(), time.monotonic(), time.monotonic()
        while duration <= 0 or time.monotonic() - start < duration:
            await asyncio.sleep(random.expovariate(rps))
            key = random.choices([t for t, _ in TENANTS], [w for _, w in TENANTS])[0]
            t = asyncio.create_task(one(client, url, key, random.choice(prompts), stats, max_tokens))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
            if time.monotonic() - last_print > 10:
                n = stats["_n"] or 1
                print(f"[{time.strftime('%H:%M:%S')}] in-flight={len(tasks)} avg_latency={stats['_lat'] / n:.2f}s "
                      + " ".join(f"{k}={v}" for k, v in sorted(stats.items()) if not k.startswith("_")), flush=True)
                stats.clear()
                last_print = time.monotonic()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rps", type=float, default=6)
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = forever")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.rps, a.duration, a.max_tokens))

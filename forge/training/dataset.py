"""Dataset preparation for supervised fine-tuning.

Accepts JSONL in either shape:
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
    {"prompt": "...", "response": "..."}            (converted to messages)
and produces a validated, de-duplicated, deterministic train/val split plus a content
hash so every trained model has exact data lineage in the registry.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

ROLES = {"system", "user", "assistant"}


@dataclass
class PreparedDataset:
    train: list[dict]
    val: list[dict]
    sha256: str
    stats: dict


def _to_messages(row: dict, lineno: int) -> list[dict]:
    if "messages" in row:
        msgs = row["messages"]
    elif "prompt" in row and "response" in row:
        msgs = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}]
        if row.get("system"):
            msgs.insert(0, {"role": "system", "content": row["system"]})
    else:
        raise ValueError(f"line {lineno}: expected 'messages' or 'prompt'/'response'")
    if not msgs or msgs[-1]["role"] != "assistant":
        raise ValueError(f"line {lineno}: conversation must end with an assistant turn")
    for m in msgs:
        if m.get("role") not in ROLES or not isinstance(m.get("content"), str) or not m["content"].strip():
            raise ValueError(f"line {lineno}: invalid message {m!r}")
    return msgs


def prepare(path: str | Path, val_fraction: float = 0.1, seed: int = 42) -> PreparedDataset:
    raw = Path(path).read_bytes()
    seen, rows, dupes = set(), [], 0
    for i, line in enumerate(raw.decode().splitlines(), 1):
        if not line.strip():
            continue
        msgs = _to_messages(json.loads(line), i)
        key = hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        rows.append({"messages": msgs})
    if len(rows) < 2:
        raise ValueError("need at least 2 unique examples")
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    chars = [sum(len(m["content"]) for m in r["messages"]) for r in rows]
    return PreparedDataset(
        train=rows[n_val:], val=rows[:n_val], sha256=hashlib.sha256(raw).hexdigest(),
        stats={"examples": len(rows), "duplicates_removed": dupes, "train": len(rows) - n_val, "val": n_val,
               "avg_chars": sum(chars) / len(chars), "max_chars": max(chars),
               "est_avg_tokens": int(sum(chars) / len(chars) / 3.8)},
    )

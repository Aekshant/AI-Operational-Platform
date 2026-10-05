"""Compare two evaluated versions and decide whether the candidate should ship.

The point is not a leaderboard. It's an engineering decision with explicit budgets:
"quality improved, but latency increased 21%: should this ship?" becomes a policy
the team agreed on, with reasons a human can audit.
"""

from __future__ import annotations

from dataclasses import dataclass

# metric -> (label, higher_is_better, format)
METRICS = {
    "accuracy": ("Accuracy", True, "{:.1%}"),
    "token_f1": ("Token F1", True, "{:.3f}"),
    "ttft_p95_s": ("TTFT p95", False, "{:.0f}ms"),
    "latency_p95_s": ("P95 latency", False, "{:.0f}ms"),
    "tokens_per_s": ("Tokens/sec", True, "{:.0f}"),
    "avg_output_tokens": ("Avg output tokens", False, "{:.0f}"),
    "cost_per_request": ("Cost/request", False, "${:.5f}"),
    "error_rate": ("Error rate", False, "{:.1%}"),
}
_MS = {"ttft_p95_s", "latency_p95_s"}


@dataclass
class ShipPolicy:
    max_quality_regression: float = 0.01   # absolute accuracy points we tolerate losing
    min_quality_gain_to_justify: float = 0.02
    max_latency_regression: float = 0.20   # relative p95 latency increase
    max_ttft_regression: float = 0.25
    max_cost_regression: float = 0.25
    max_error_rate: float = 0.02


def _fmt(metric: str, v: float) -> str:
    return METRICS[metric][2].format(v * 1000 if metric in _MS else v)


def compare(base: dict, cand: dict, policy: ShipPolicy | None = None) -> dict:
    """base/cand are registry entries ({'version':..., 'metrics': {...}})."""
    policy = policy or ShipPolicy()
    bm, cm = base["metrics"], cand["metrics"]
    rows = []
    for k, (label, higher_better, _) in METRICS.items():
        if k not in bm or k not in cm:
            continue
        b, c = float(bm[k]), float(cm[k])
        rel = (c - b) / b if b else 0.0
        better = (c > b) if higher_better else (c < b)
        rows.append({"metric": k, "label": label, "base": _fmt(k, b), "candidate": _fmt(k, c),
                     "delta_pct": round(rel * 100, 1), "better": better if c != b else None})

    reasons, blockers, tradeoffs = [], [], []
    dq = cm.get("accuracy", 0) - bm.get("accuracy", 0)
    if dq < -policy.max_quality_regression:
        blockers.append(f"quality regressed {dq * 100:+.1f} pts (tolerance {policy.max_quality_regression * 100:.0f})")
    if cm.get("error_rate", 0) > policy.max_error_rate:
        blockers.append(f"error rate {cm['error_rate']:.1%} exceeds {policy.max_error_rate:.0%}")
    for k, budget, name in (("latency_p95_s", policy.max_latency_regression, "p95 latency"),
                            ("ttft_p95_s", policy.max_ttft_regression, "TTFT p95"),
                            ("cost_per_request", policy.max_cost_regression, "cost/request")):
        if bm.get(k) and cm.get(k):
            rel = (cm[k] - bm[k]) / bm[k]
            if rel > budget:
                tradeoffs.append(f"{name} increased {rel:.0%} (budget {budget:.0%})")

    if blockers:
        decision = "BLOCK"
        reasons = blockers
    elif tradeoffs and dq >= policy.min_quality_gain_to_justify:
        decision = "NEEDS_REVIEW"
        reasons = [f"quality improved {dq * 100:+.1f} pts, but " + "; ".join(tradeoffs)]
    elif tradeoffs:
        decision = "BLOCK"
        reasons = tradeoffs + [f"quality gain {dq * 100:+.1f} pts does not justify the regression"]
    else:
        decision = "SHIP_CANARY"
        reasons = [f"quality {dq * 100:+.1f} pts; latency, TTFT and cost within budget"]

    headline = f"{base['version']} → {cand['version']}: {decision}. " + " ".join(reasons)
    return {"base": base["version"], "candidate": cand["version"], "rows": rows, "decision": decision,
            "reasons": reasons, "headline": headline}

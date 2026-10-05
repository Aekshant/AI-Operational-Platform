"""Cost attribution: "why did inference cost go up?"

Cost = Requests x Tokens/request x Price/token. Taking logs makes the change additive:

    ln(C1/C0) = ln(R1/R0) + ln(T1/T0) + ln(P1/P0)

so each factor's share of the total change is exact and sums to 100% (the plain
percentage deltas don't add up once the change is large). Prompt vs completion tokens and
per-version traffic mix are broken out so the answer names a concrete cause.
"""

from __future__ import annotations

import math

from forge.agent.prom import Prom


def explain_cost_change(prom: Prom, deployment: str | None = None, window: str = "1h",
                        baseline_offset: str = "1d") -> dict:
    sel = f'deployment="{deployment}"' if deployment else ""

    def total(metric: str, extra: str = "", offset: str = "") -> float:
        f = ",".join(x for x in (sel, extra) if x)
        off = f" offset {offset}" if offset else ""
        return prom.scalar(f"sum(increase({metric}{{{f}}}[{window}]{off}))", 0.0) or 0.0

    now = {"cost": total("forge_cost_usd_total"), "requests": total("forge_requests_total"),
           "prompt": total("forge_tokens_total", 'kind="prompt"'),
           "completion": total("forge_tokens_total", 'kind="completion"')}
    base = {"cost": total("forge_cost_usd_total", offset=baseline_offset),
            "requests": total("forge_requests_total", offset=baseline_offset),
            "prompt": total("forge_tokens_total", 'kind="prompt"', baseline_offset),
            "completion": total("forge_tokens_total", 'kind="completion"', baseline_offset)}
    if min(base["cost"], base["requests"], now["requests"]) <= 0:
        return {"ok": False, "reason": "not enough traffic in one of the windows to attribute cost",
                "current": now, "baseline": base}

    def per_req(d, k):
        return d[k] / d["requests"]

    t0 = per_req(base, "prompt") + per_req(base, "completion")
    t1 = per_req(now, "prompt") + per_req(now, "completion")
    p0 = base["cost"] / (base["requests"] * t0) if t0 else 0
    p1 = now["cost"] / (now["requests"] * t1) if t1 else 0
    factors = {"traffic": now["requests"] / base["requests"], "tokens_per_request": t1 / t0 if t0 else 1,
               "unit_price": p1 / p0 if p0 else 1}
    total_log = math.log(now["cost"] / base["cost"])
    shares = {k: (math.log(v) / total_log if total_log else 0.0) for k, v in factors.items()}
    driver = max(shares, key=lambda k: shares[k] if total_log > 0 else -shares[k])

    mix_now = prom.by(f"sum by (version) (increase(forge_requests_total{{{sel}}}[{window}]))", "version")
    mix_base = prom.by(f"sum by (version) (increase(forge_requests_total{{{sel}}}[{window}] offset "
                       f"{baseline_offset}))", "version")
    out_tok_now = prom.by(
        f'sum by (version) (increase(forge_tokens_total{{{sel}{"," if sel else ""}kind="completion"}}[{window}]))'
        f" / sum by (version) (increase(forge_requests_total{{{sel}}}[{window}]))", "version")

    pct = lambda r: round((r - 1) * 100, 1)  # noqa: E731
    change = now["cost"] / base["cost"]
    summary = (
        f"Cost {pct(change):+}% vs baseline ({base['cost']:.4f} → {now['cost']:.4f} USD per {window}). "
        f"Traffic {pct(factors['traffic']):+}%, tokens/request {pct(factors['tokens_per_request']):+}% "
        f"(completion {per_req(base, 'completion'):.0f} → {per_req(now, 'completion'):.0f}, "
        f"prompt {per_req(base, 'prompt'):.0f} → {per_req(now, 'prompt'):.0f}), "
        f"unit price {pct(factors['unit_price']):+}%. "
        f"Primary driver: {driver.replace('_', ' ')} ({shares[driver]:.0%} of the change)."
    )
    return {
        "ok": True, "summary": summary, "cost_change_pct": pct(change), "factors_pct":
            {k: pct(v) for k, v in factors.items()}, "share_of_change": {k: round(v, 3) for k, v in shares.items()},
        "primary_driver": driver,
        "tokens_per_request": {"baseline": {"prompt": per_req(base, "prompt"),
                                            "completion": per_req(base, "completion")},
                               "current": {"prompt": per_req(now, "prompt"),
                                           "completion": per_req(now, "completion")}},
        "traffic_mix": {"current": mix_now, "baseline": mix_base}, "completion_tokens_by_version": out_tok_now,
        "window": window, "baseline_offset": baseline_offset,
    }

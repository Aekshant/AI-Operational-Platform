import math

import pytest

from forge.agent.cost import explain_cost_change
from forge.evaluation.compare import compare
from forge.evaluation.scorers import contains, exact_match, token_f1
from forge.training.dataset import prepare


def test_scorers():
    assert exact_match("The Nitrogen.", "nitrogen") == 1
    assert contains("It is mostly nitrogen, applied early.", "Nitrogen") == 1
    assert 0 < token_f1("drip irrigation system", "drip irrigation") < 1


def _mv(v, **m):
    return {"version": v, "metrics": m}


def test_ship_gate_decisions():
    base = _mv("v1", accuracy=0.81, latency_p95_s=0.42, ttft_p95_s=0.2, cost_per_request=0.002, error_rate=0)
    better_slower = _mv("v2", accuracy=0.87, latency_p95_s=0.51, ttft_p95_s=0.21, cost_per_request=0.0022,
                        error_rate=0)
    r = compare(base, better_slower)
    assert r["decision"] == "NEEDS_REVIEW" and "21%" in r["reasons"][0]
    assert compare(base, _mv("v3", accuracy=0.75, latency_p95_s=0.4, ttft_p95_s=0.2, cost_per_request=0.002,
                             error_rate=0))["decision"] == "BLOCK"
    assert compare(base, _mv("v4", accuracy=0.83, latency_p95_s=0.43, ttft_p95_s=0.2, cost_per_request=0.002,
                             error_rate=0))["decision"] == "SHIP_CANARY"


def test_dataset_prep_dedupes_and_hashes(tmp_path):
    p = tmp_path / "d.jsonl"
    row = '{"prompt": "q%d", "response": "a"}\n'
    p.write_text("".join(row % i for i in range(10)) + row % 0)
    ds = prepare(p)
    assert ds.stats["duplicates_removed"] == 1 and ds.stats["examples"] == 10
    assert len(ds.sha256) == 64 and len(ds.val) == 1


def test_dataset_rejects_bad_rows(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    with pytest.raises(ValueError, match="assistant"):
        prepare(p)


class CostProm:
    """Baseline: 1000 req, 200 prompt + 210 completion tok. Now: traffic +8%, completion 390."""

    def scalar(self, q, default=None):
        base = "offset" in q
        req = 1000 if base else 1080
        if "forge_cost_usd_total" in q:
            return req * ((200 + (210 if base else 390)) / 1000) * 0.0005
        if "forge_requests_total" in q:
            return req
        if 'kind="prompt"' in q:
            return req * 200
        if 'kind="completion"' in q:
            return req * (210 if base else 390)
        return default

    def by(self, q, label):
        return {}


def test_cost_decomposition_names_tokens_as_driver():
    r = explain_cost_change(CostProm(), "qwen-farm", "1h", "1d")
    assert r["ok"] and r["primary_driver"] == "tokens_per_request"
    assert r["factors_pct"]["traffic"] == pytest.approx(8.0)
    assert math.isclose(sum(r["share_of_change"].values()), 1.0, abs_tol=1e-3)
    assert "210 → 390" in r["summary"]

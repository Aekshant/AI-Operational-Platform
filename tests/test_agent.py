"""The incident lifecycle end to end, against a fake Prometheus that tells the v3-regression story."""

import json

import pytest

from forge.agent.graph import IncidentGraph
from forge.agent.incidents import IncidentStore
from forge.agent.investigator import HeuristicInvestigator
from forge.agent.tools import Toolbox


class FakeProm:
    def __init__(self):
        self.state = "regressed"

    def _data(self):
        bad = self.state == "regressed"
        return {
            "deployment": {"ttft": {"qwen-farm": 2.9 if bad else 0.22}, "lat": {"qwen-farm": 6.0 if bad else 1.4},
                           "err": {"qwen-farm": 0.004 if bad else 0.0}},
            "version": {
                "ttft": {"v1": 0.21, "v3": 3.4} if bad else {"v1": 0.21},
                "lat": {"v1": 1.4, "v3": 6.5} if bad else {"v1": 1.4},
                "err": {"v1": 0.0, "v3": 0.01} if bad else {"v1": 0.0},
                "reqs": {"v1": 3.0, "v3": 3.0} if bad else {"v1": 6.0},
                "comp": {"v1": 390.0, "v3": 400.0} if bad else {"v1": 780.0},
                "tpot": {"v1": 0.0095, "v3": 0.011} if bad else {"v1": 0.0095},
                "util": {"v1": 40.0, "v2": 2.0, "v3": 97.0}, "mem": {"v1": 14.6, "v2": 14.0, "v3": 19.4},
                "free": {"v1": 9.4, "v2": 10.0, "v3": 4.6}, "kv": {"v1": 0.1, "v2": 0.0, "v3": 0.8},
                "running": {"v1": 4.0, "v3": 4.0}, "waiting": {"v1": 0.0, "v3": 14.0 if bad else 0.0},
            },
            "backend": {"queue": {"qwen-farm:v3": 6.0 if bad else 0.0}, "inflight": {"qwen-farm:v1": 4.0},
                        "breaker": {"qwen-farm:v3": 0.0}},
            "reason": {"rej": {}},
        }

    @staticmethod
    def _kind(q):
        for key, k in (("forge_ttft_seconds", "ttft"), ("forge_request_duration", "lat"), ('code!="200"', "err"),
                       ("forge_tpot", "tpot"), ('kind="completion"', "comp"), ("forge_requests_total", "reqs"),
                       ("GPU_UTIL", "util"), ("FB_USED", "mem"), ("FB_FREE", "free"), ("gpu_cache", "kv"),
                       ("num_requests_running", "running"), ("num_requests_waiting", "waiting"),
                       ("forge_queue_depth", "queue"), ("forge_inflight", "inflight"), ("forge_rejected", "rej"),
                       ("forge_breaker", "breaker")):
            if key in q:
                return k
        return None

    def by(self, q, label):
        return dict(self._data().get(label, {}).get(self._kind(q), {}))

    def scalar(self, q, default=None):
        return 0.0

    def query(self, q):
        return []


@pytest.fixture
def toolbox(platform):
    reg, ctl = platform
    ctl.start_canary("qwen-farm", "v3", 50, actor="ci")
    tb = Toolbox(prom=FakeProm(), controller=ctl, registry=reg)
    tb.loki.errors = lambda *a, **k: []  # no Loki in unit tests
    return tb


def test_heuristic_blames_the_canary(toolbox):
    d = HeuristicInvestigator(toolbox).investigate("qwen-farm")
    assert d.action == "rollback_model" and d.action_args["deployment"] == "qwen-farm"
    assert "v3" in d.summary and d.confidence >= 0.8
    text = " ".join(d.evidence)
    assert "start_canary" in text and "TTFT p95" in text and "GPU memory" in text and "+33%" in text
    assert len(d.tool_calls) >= 6


def test_full_incident_lifecycle_with_approval(toolbox, conn, tmp_path):
    store = IncidentStore(conn)
    g = IncidentGraph(tools=toolbox, store=store, checkpoint_path=str(tmp_path / "ckpt.db"),
                      verify_window_s=5, verify_interval_s=0)
    inc = store.create("TTFT SLO violation", {"deployment": "qwen-farm"})
    g.start(inc["id"], "qwen-farm", {"alertname": "InferenceTTFTSLOViolation"})

    cur = store.get(inc["id"])
    assert cur["state"] == "AWAITING_APPROVAL"
    assert cur["proposed_action"]["action"] == "rollback_model"
    assert g.state(inc["id"])["next"] == ["await_approval"]
    assert toolbox.ctl.get("qwen-farm")["canary_version"] == "v3"  # nothing executed before approval

    toolbox.prom.state = "healthy"  # what production looks like once v3 is out of the path
    g2 = IncidentGraph(tools=toolbox, store=store, checkpoint_path=str(tmp_path / "ckpt.db"),
                       verify_window_s=5, verify_interval_s=0)  # a new process: resume from the checkpoint
    g2.resume(inc["id"], approved=True, by="oncall@forge")

    final = store.get(inc["id"])
    assert final["state"] == "RESOLVED"
    phases = [t["phase"] for t in final["timeline"] if t["phase"]]
    assert phases == ["INCIDENT", "INVESTIGATING", "DIAGNOSIS_READY", "AWAITING_APPROVAL", "EXECUTING",
                      "VERIFYING", "RESOLVED"]
    dep = toolbox.ctl.get("qwen-farm")
    assert dep["canary_version"] is None and dep["stable_version"] == "v1"
    assert toolbox.reg.get("qwen-farm", "v3")["stage"] == "rejected"
    last = toolbox.ctl.events("qwen-farm")[0]
    assert last["action"] == "rollback" and "oncall@forge" in last["actor"]


def test_rejection_changes_nothing(toolbox, conn, tmp_path):
    store = IncidentStore(conn)
    g = IncidentGraph(tools=toolbox, store=store, checkpoint_path=str(tmp_path / "c.db"), verify_window_s=1)
    inc = store.create("x", {})
    g.start(inc["id"], "qwen-farm")
    g.resume(inc["id"], approved=False, by="oncall", comment="investigating manually")
    assert store.get(inc["id"])["state"] == "REJECTED"
    assert toolbox.ctl.get("qwen-farm")["canary_version"] == "v3"


def test_execute_refuses_cross_deployment_action(toolbox, conn, tmp_path):
    """A (possibly prompt-injected) proposal targeting another deployment never executes."""
    store = IncidentStore(conn)
    g = IncidentGraph(tools=toolbox, store=store, checkpoint_path=str(tmp_path / "c.db"), verify_window_s=1)
    inc = store.create("x", {})
    g.start(inc["id"], "qwen-farm", proposal={"action": "rollback_model",
                                               "args": {"deployment": "payments-model", "reason": "x"}})
    with pytest.raises(ValueError, match="different deployment"):
        g.resume(inc["id"], approved=True, by="oncall")
    assert toolbox.ctl.get("qwen-farm")["canary_version"] == "v3"


def test_tool_errors_are_returned_not_raised(toolbox):
    out = json.loads(toolbox.call_json("get_deployment_status", {"deployment": "nope"}))
    assert "error" in out

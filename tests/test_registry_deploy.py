import pytest

from forge.deploy import DeployError
from forge.registry import RegistryError


def test_versions_are_immutable(registry):
    registry.register("m", "v1", "base")
    with pytest.raises(RegistryError):
        registry.register("m", "v1", "base")


def test_single_production_version(registry):
    registry.register("m", "v1", "base")
    registry.register("m", "v2", "base")
    registry.transition("m", "v1", "production")
    registry.transition("m", "v2", "production")
    assert registry.get("m", "v1")["stage"] == "archived"
    assert registry.production("m")["version"] == "v2"
    assert [h["to_stage"] for h in registry.history("m")][-2:] == ["archived", "production"]


def test_illegal_transition(registry):
    registry.register("m", "v1", "base")
    registry.transition("m", "v1", "production")
    with pytest.raises(RegistryError):
        registry.transition("m", "v1", "candidate")


def test_canary_routing_and_promote(platform):
    reg, ctl = platform
    d = ctl.start_canary("qwen-farm", "v2", 20)
    assert d["status"] == "canary" and reg.get("qwen-farm", "v2")["stage"] == "staging"
    route = ctl.routing()["qwen-farm"]
    assert {(b["version"], b["weight"]) for b in route["backends"]} == {("v1", 80), ("v2", 20)}
    assert route["fallback"]["version"] == "v1"
    ctl.promote("qwen-farm")
    assert ctl.get("qwen-farm")["stable_version"] == "v2"
    assert reg.production("qwen-farm")["version"] == "v2"
    assert reg.get("qwen-farm", "v1")["stage"] == "archived"


def test_rollback_canary_rejects_version(platform):
    reg, ctl = platform
    ctl.start_canary("qwen-farm", "v3", 50)
    d = ctl.rollback("qwen-farm", reason="ttft regression")
    assert d["canary_version"] is None and d["stable_version"] == "v1"
    assert reg.get("qwen-farm", "v3")["stage"] == "rejected"
    assert ctl.events("qwen-farm")[0]["action"] == "rollback"


def test_rollback_stable_to_previous(platform):
    reg, ctl = platform
    ctl.start_canary("qwen-farm", "v2", 10)
    ctl.promote("qwen-farm")
    d = ctl.rollback("qwen-farm", reason="bad promote")
    assert d["stable_version"] == "v1"
    assert reg.production("qwen-farm")["version"] == "v1"


def test_idempotency_key_applies_once(platform):
    _, ctl = platform
    ctl.start_canary("qwen-farm", "v2", 10)
    a = ctl.rollback("qwen-farm", reason="x", idempotency_key="inc-1:rollback_model")
    b = ctl.rollback("qwen-farm", reason="x", idempotency_key="inc-1:rollback_model")
    assert a == b
    assert sum(e["action"] == "rollback" for e in ctl.events("qwen-farm")) == 1


def test_optimistic_concurrency(platform):
    _, ctl = platform
    rev = ctl.get("qwen-farm")["revision"]
    ctl.start_canary("qwen-farm", "v2", 10, expected_revision=rev)
    with pytest.raises(DeployError, match="revision conflict"):
        ctl.set_weight("qwen-farm", 50, expected_revision=rev)


def test_cannot_canary_without_serving_url(platform):
    reg, ctl = platform
    reg.register("qwen-farm", "v9", "base")
    with pytest.raises(DeployError, match="serving_url"):
        ctl.start_canary("qwen-farm", "v9", 10)


def test_rejected_version_needs_requalification(platform):
    reg, ctl = platform
    ctl.start_canary("qwen-farm", "v3", 50)
    ctl.rollback("qwen-farm", reason="regression")
    with pytest.raises(DeployError, match="re-qualify"):
        ctl.start_canary("qwen-farm", "v3", 10)
    reg.transition("qwen-farm", "v3", "candidate", reason="fixed max_num_seqs")
    assert ctl.start_canary("qwen-farm", "v3", 10)["canary_version"] == "v3"

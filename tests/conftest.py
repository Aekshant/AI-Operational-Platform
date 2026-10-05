import os
import tempfile

os.environ.setdefault("FORGE_HOME", tempfile.mkdtemp(prefix="forge-test-"))
os.environ["FORGE_AGENT_LLM"] = "heuristic"

import pytest  # noqa: E402

from forge import db  # noqa: E402
from forge.deploy import DeployController  # noqa: E402
from forge.registry import Registry  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "forge.db")


@pytest.fixture
def registry(conn):
    return Registry(conn)


@pytest.fixture
def controller(conn, registry):
    return DeployController(conn, registry)


@pytest.fixture
def platform(registry, controller):
    """qwen-farm with v1 (prod, stable) and v2, v3 registered with serving URLs and eval metrics."""
    for v, acc, p95 in (("v1", 0.80, 1.2), ("v2", 0.88, 1.4), ("v3", 0.89, 1.5)):
        registry.register("qwen-farm", v, "Qwen/Qwen2.5-1.5B-Instruct", serving_url=f"http://sim-{v}:8000",
                          metrics={"accuracy": acc, "latency_p95_s": p95, "ttft_p95_s": 0.2,
                                   "cost_per_request": 0.0001, "error_rate": 0.0, "cost_per_1k_tokens": 0.0005})
    controller.create("qwen-farm", "qwen-farm", "v1")
    return registry, controller

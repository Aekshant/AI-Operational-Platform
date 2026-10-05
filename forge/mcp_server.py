"""MCP server: exposes ForgeAI's operations toolbox to any MCP client (Claude Code, Claude Desktop, IDEs).

Read tools are exposed directly. Write actions are NOT: an external agent can only call
`request_action`, which files an incident carrying the proposal into the same approval gate
the in-house agent uses. One HITL policy covers every agent.

    claude mcp add forgeai -- forge serve mcp
"""

from __future__ import annotations

import inspect
import json
import os

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from forge.agent.investigator import ACTIONS
from forge.agent.tools import Toolbox
from forge.config import settings

server = MCPServer(
    "forgeai",
    instructions="Operate the ForgeAI model platform. Read tools observe production. To change production, "
                 "call request_action; a human must approve it in ForgeAI before anything executes.",
)
_tb: Toolbox | None = None


def toolbox() -> Toolbox:
    global _tb
    if _tb is None:
        _tb = Toolbox()
    return _tb


def _register_read_tools() -> None:
    names = ["get_slo_status", "get_service_health", "get_model_metrics", "get_gpu_metrics", "get_queue_status",
             "get_recent_errors", "get_deployment_status", "compare_model_versions", "search_slow_traces",
             "get_trace"]
    for name in names:
        method = getattr(Toolbox, name)

        def make(n=name, m=method):
            def fn(*args, **kwargs):
                return json.loads(json.dumps(getattr(toolbox(), n)(*args, **kwargs), default=str))

            sig = inspect.signature(m)
            fn.__signature__ = sig.replace(parameters=list(sig.parameters.values())[1:])
            fn.__annotations__ = {k: v for k, v in m.__annotations__.items() if k != "self"}
            return fn

        server.add_tool(make(), name=name, description=toolbox().tools[name].description,
                        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))

    def explain_cost_change(deployment: str = "qwen-farm", window: str = "1h", baseline_offset: str = "1d") -> dict:
        """Attribute a serving-cost change to traffic, tokens per request and unit price."""
        return toolbox().explain_cost(deployment, window, baseline_offset)

    server.add_tool(explain_cost_change, annotations=ToolAnnotations(read_only_hint=True))


_register_read_tools()


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False))
def request_action(deployment: str, action: str, reason: str, weight: int | None = None,
                   replicas: int | None = None) -> dict:
    """Propose a production change (rollback_model, pause_rollout, set_canary_weight, scale_inference).
    Opens an incident in AWAITING_APPROVAL; nothing executes until a human approves it in ForgeAI."""
    if action not in ACTIONS or action == "none":
        return {"error": f"action must be one of {[a for a in ACTIONS if a != 'none']}"}
    args = {"deployment": deployment, "reason": reason, "weight": weight, "replicas": replicas}
    r = httpx.post(f"{settings().controlplane_url}/api/incidents",
                   headers={"Authorization": f"Bearer {os.environ.get('FORGE_API_KEY', 'dev-admin')}"},
                   json={"deployment": deployment, "title": f"MCP request: {action} ({reason[:80]})",
                         "severity": "warning",
                         "proposal": {"action": action, "args": {k: v for k, v in args.items() if v is not None},
                                      "rationale": reason}}, timeout=10)
    r.raise_for_status()
    inc = r.json()
    return {"incident_id": inc["id"], "status": "awaiting human approval",
            "approve_with": f"forge incident approve {inc['id']}"}

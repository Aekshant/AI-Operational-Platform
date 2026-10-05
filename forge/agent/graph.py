"""The incident lifecycle as a durable LangGraph state machine.

    INCIDENT -> INVESTIGATING -> DIAGNOSIS_READY -> AWAITING_APPROVAL -> EXECUTING -> VERIFYING -> RESOLVED
                                        |                  |                                |
                                        v                  v                                v
                                    ESCALATED           REJECTED                      VERIFY_FAILED

Durability: state is checkpointed to SQLite after every node. `interrupt()` suspends the run at the
approval gate; the process can restart and the run resumes from the checkpoint when a human answers.
Execution safety: the action must be in the allowlist, arguments are re-validated, the target
deployment must match the incident, and the write uses an idempotency key derived from the incident
id, so a retried EXECUTING step can never roll back twice.
"""

from __future__ import annotations

import logging
import operator
import sqlite3
import time
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from forge.agent.incidents import IncidentStore
from forge.agent.investigator import ACTIONS, investigate
from forge.agent.tools import Toolbox
from forge.config import settings

log = logging.getLogger(__name__)


class IncidentState(TypedDict, total=False):
    incident_id: str
    deployment: str
    alert: dict
    phase: str
    diagnosis: dict
    decision: dict
    action_result: dict
    verification: dict
    log: Annotated[list[str], operator.add]


class IncidentGraph:
    def __init__(self, tools: Toolbox | None = None, store: IncidentStore | None = None,
                 checkpoint_path: str | None = None, verify_window_s: float | None = None,
                 verify_interval_s: float = 5.0):
        self.tools = tools or Toolbox()
        self.store = store or IncidentStore()
        self.verify_window_s = verify_window_s if verify_window_s is not None else settings().verify_window_s
        self.verify_interval_s = verify_interval_s
        path = checkpoint_path or str(settings().home / "agent_checkpoints.db")
        self.saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        self.graph = self._build().compile(checkpointer=self.saver)

    # ------------------------------------------------------------------ nodes
    def _phase(self, s: IncidentState, phase: str, note: str, **fields) -> None:
        self.store.update(s["incident_id"], phase=phase, note=note, **fields)

    def intake(self, s: IncidentState) -> dict:
        self._phase(s, "INVESTIGATING", f"Investigating {s['deployment']}")
        return {"phase": "INVESTIGATING", "log": ["intake"]}

    def investigate(self, s: IncidentState) -> dict:
        if s.get("diagnosis"):  # proposal filed by an external agent (MCP): skip straight to the gate
            d = s["diagnosis"]
            self._phase(s, "DIAGNOSIS_READY", d["summary"], diagnosis=d)
            return {"phase": "DIAGNOSIS_READY", "log": ["external proposal"]}
        d = investigate(self.tools, s["deployment"], s.get("alert")).to_dict()
        self._phase(s, "DIAGNOSIS_READY", d["summary"], diagnosis=d)
        return {"phase": "DIAGNOSIS_READY", "diagnosis": d, "log": [f"diagnosed by {d['investigator']}"]}

    def route_after_diagnosis(self, s: IncidentState) -> str:
        return "propose" if s["diagnosis"]["action"] != "none" else "escalate"

    def propose(self, s: IncidentState) -> dict:
        d = s["diagnosis"]
        proposal = {"action": d["action"], "args": d["action_args"], "rationale": d["rationale"]}
        self._phase(s, "AWAITING_APPROVAL", f"Recommends {d['action']} {d['action_args']}", proposed_action=proposal)
        return {"phase": "AWAITING_APPROVAL", "log": ["proposed"]}

    def await_approval(self, s: IncidentState) -> dict:
        # LangGraph re-runs an interrupted node from the top on resume, so this node must have
        # no side effects before interrupt(). Recording the proposal lives in `propose`.
        d = s["diagnosis"]
        proposal = {"action": d["action"], "args": d["action_args"], "rationale": d["rationale"]}
        decision = interrupt({"incident_id": s["incident_id"], "proposal": proposal, "summary": d["summary"]})
        approved = bool(decision.get("approved"))
        who = decision.get("by", "unknown")
        if not approved:
            self._phase(s, "REJECTED", f"Rejected by {who}: {decision.get('comment', '')}")
        return {"decision": decision, "phase": "EXECUTING" if approved else "REJECTED",
                "log": [f"{'approved' if approved else 'rejected'} by {who}"]}

    def route_after_approval(self, s: IncidentState) -> str:
        return "execute" if s["decision"].get("approved") else END

    def execute(self, s: IncidentState) -> dict:
        d = s["diagnosis"]
        action, args = d["action"], dict(d["action_args"])
        if action not in ACTIONS or action == "none" or self.tools.tools[action].risk != "write":
            raise ValueError(f"action {action} not allowed")
        if args.get("deployment") != s["deployment"]:
            raise ValueError("action targets a different deployment than the incident")
        allowed = set(self.tools.tools[action].params)
        args = {k: v for k, v in args.items() if k in allowed}
        if action == "rollback_model":
            args.setdefault("reason", d["summary"][:200])
        self._phase(s, "EXECUTING", f"Executing {action} {args} (approved by {s['decision'].get('by')})")
        result = self.tools.call(action, args, actor=f"agent/approved-by:{s['decision'].get('by')}",
                                 idempotency_key=f"{s['incident_id']}:{action}")
        self._phase(s, "VERIFYING", f"{action} applied (revision {result.get('revision')}); verifying SLOs")
        return {"action_result": result, "phase": "VERIFYING", "log": [f"executed {action}"]}

    def verify(self, s: IncidentState) -> dict:
        """Recovery = SLOs within target on N consecutive checks within the verification window."""
        deadline, need, streak, last = time.time() + self.verify_window_s, 3, 0, {}
        while time.time() < deadline:
            try:
                last = self.tools.get_slo_status().get(s["deployment"], {})
            except Exception as e:
                last = {"error": str(e)}
            healthy = "violations" in last and not last["violations"]
            streak = streak + 1 if healthy else 0
            if streak >= need:
                note = (f"Recovery confirmed: TTFT p95 {last.get('ttft_p95_s')}s, "
                        f"error rate {last.get('error_rate')} within SLO for {need} consecutive checks")
                self._phase(s, "RESOLVED", note)
                return {"verification": {"ok": True, **last}, "phase": "RESOLVED", "log": ["verified"]}
            time.sleep(self.verify_interval_s)
        self._phase(s, "VERIFY_FAILED", f"SLOs not restored within {self.verify_window_s:.0f}s: {last}")
        return {"verification": {"ok": False, **last}, "phase": "VERIFY_FAILED", "log": ["verify failed"]}

    def escalate(self, s: IncidentState) -> dict:
        self._phase(s, "ESCALATED", "No safe automated action; paging on-call with the diagnosis attached")
        return {"phase": "ESCALATED", "log": ["escalated"]}

    def _build(self) -> StateGraph:
        g = StateGraph(IncidentState)
        for name in ("intake", "investigate", "propose", "await_approval", "execute", "verify", "escalate"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "intake")
        g.add_edge("intake", "investigate")
        g.add_conditional_edges("investigate", self.route_after_diagnosis, ["propose", "escalate"])
        g.add_edge("propose", "await_approval")
        g.add_conditional_edges("await_approval", self.route_after_approval, ["execute", END])
        g.add_edge("execute", "verify")
        g.add_edge("verify", END)
        g.add_edge("escalate", END)
        return g

    # ------------------------------------------------------------------ API
    def start(self, incident_id: str, deployment: str, alert: dict | None = None,
              proposal: dict | None = None) -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": incident_id}}
        init: IncidentState = {"incident_id": incident_id, "deployment": deployment, "alert": alert or {},
                               "phase": "INCIDENT", "log": []}
        if proposal:
            init["diagnosis"] = {"summary": f"External request: {proposal['action']}", "root_cause": "n/a",
                                 "evidence": [], "confidence": 0.0, "suspected_change": None,
                                 "action": proposal["action"], "action_args": proposal["args"],
                                 "rationale": proposal.get("rationale", ""), "investigator": "external (MCP)",
                                 "tool_calls": []}
        return self.graph.invoke(init, cfg)

    def resume(self, incident_id: str, approved: bool, by: str, comment: str = "") -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": incident_id}}
        snap = self.graph.get_state(cfg)
        if not snap.next or "await_approval" not in snap.next:
            raise RuntimeError(f"incident {incident_id} is not awaiting approval (next={snap.next})")
        return self.graph.invoke(Command(resume={"approved": approved, "by": by, "comment": comment}), cfg)

    def state(self, incident_id: str) -> dict:
        snap = self.graph.get_state({"configurable": {"thread_id": incident_id}})
        return {"values": snap.values, "next": list(snap.next)}

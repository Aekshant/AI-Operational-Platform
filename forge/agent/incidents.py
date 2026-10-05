"""Incident records: the human-facing projection of the agent's state machine."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from forge import db

PHASES = ["INCIDENT", "INVESTIGATING", "DIAGNOSIS_READY", "AWAITING_APPROVAL", "EXECUTING", "VERIFYING",
          "RESOLVED", "REJECTED", "ESCALATED", "VERIFY_FAILED"]
OPEN_PHASES = PHASES[:6]


class IncidentStore:
    def __init__(self, conn=None):
        self.conn = conn or db.connect()

    def create(self, title: str, labels: dict, severity: str = "critical", source: str = "manual") -> dict:
        iid = f"inc-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        now = time.time()
        with db.tx(self.conn) as c:
            c.execute("INSERT INTO incidents(id, title, severity, state, source, labels, diagnosis, proposed_action,"
                      " timeline, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (iid, title, severity, "INCIDENT", source, json.dumps(labels), None, None,
                       json.dumps([{"at": now, "phase": "INCIDENT", "note": title}]), now, now))
        return self.get(iid)

    def find_open(self, fingerprint: str) -> dict | None:
        """Alert de-duplication: one open incident per fingerprint."""
        for r in self.conn.execute("SELECT * FROM incidents WHERE state IN (%s) ORDER BY created_at DESC"
                                   % ",".join("?" * len(OPEN_PHASES)), OPEN_PHASES):
            d = db.row(r, ("labels",))
            if d["labels"].get("fingerprint") == fingerprint:
                return d
        return None

    def get(self, iid: str) -> dict:
        r = db.row(self.conn.execute("SELECT * FROM incidents WHERE id=?", (iid,)).fetchone(),
                   ("labels", "diagnosis", "proposed_action", "timeline"))
        if not r:
            raise KeyError(iid)
        return r

    def list(self, limit: int = 50) -> list[dict]:
        return [db.row(r, ("labels", "diagnosis", "proposed_action", "timeline")) for r in self.conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?", (limit,))]

    def update(self, iid: str, phase: str | None = None, note: str | None = None, **fields: Any) -> None:
        with db.tx(self.conn) as c:
            cur = db.row(c.execute("SELECT timeline FROM incidents WHERE id=?", (iid,)).fetchone(), ("timeline",))
            timeline = cur["timeline"] if cur else []
            if phase or note:
                timeline.append({"at": time.time(), "phase": phase, "note": note})
            sets, args = ["timeline=?", "updated_at=?"], [json.dumps(timeline, default=str), time.time()]
            if phase:
                sets.append("state=?")
                args.append(phase)
            for k, v in fields.items():
                sets.append(f"{k}=?")
                args.append(json.dumps(v, default=str))
            c.execute(f"UPDATE incidents SET {', '.join(sets)} WHERE id=?", (*args, iid))

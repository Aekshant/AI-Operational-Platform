"""Deployment controller: the desired-state store for model rollouts.

A deployment maps a stable model version and an optional canary version to traffic
weights. The gateway polls `routing()` and converges on it (eventual consistency,
bounded by FORGE_ROUTING_REFRESH_S). Every mutation:
  * bumps a monotonically increasing `revision` (optimistic concurrency via `expected_revision`)
  * is idempotent when the caller supplies an idempotency key
  * is written to `deployment_events` so the agent can answer "what changed?"
"""

from __future__ import annotations

import json
import time
from typing import Any

from forge import db
from forge.registry import Registry


class DeployError(Exception):
    pass


class DeployController:
    def __init__(self, conn=None, registry: Registry | None = None):
        self.conn = conn or db.connect()
        self.registry = registry or Registry(self.conn)

    # ---------- reads ----------
    def get(self, name: str) -> dict[str, Any]:
        r = db.row(self.conn.execute("SELECT * FROM deployments WHERE name=?", (name,)).fetchone())
        if not r:
            raise DeployError(f"unknown deployment {name}")
        return r

    def list(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM deployments ORDER BY name")]

    def events(self, name: str | None = None, since: float = 0, limit: int = 50) -> list[dict[str, Any]]:
        q = "SELECT * FROM deployment_events WHERE at >= ?"
        args: list[Any] = [since]
        if name:
            q += " AND deployment=?"
            args.append(name)
        rows = self.conn.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [db.row(r, ("detail",)) for r in rows]

    def routing(self) -> dict[str, Any]:
        """The gateway's view: per deployment, the weighted backends plus a fallback."""
        out = {}
        for d in self.list():
            backends = []
            stable = self._backend(d["model"], d["stable_version"]) if d["stable_version"] else None
            canary = self._backend(d["model"], d["canary_version"]) if d["canary_version"] else None
            w = d["canary_weight"] if canary else 0
            if stable:
                backends.append({**stable, "weight": 100 - w, "role": "stable"})
            if canary and w > 0:
                backends.append({**canary, "weight": w, "role": "canary"})
            out[d["name"]] = {"revision": d["revision"], "status": d["status"], "backends": backends,
                              "fallback": stable}
        return out

    def _backend(self, model: str, version: str) -> dict[str, Any]:
        mv = self.registry.get(model, version)
        if not mv.get("serving_url"):
            raise DeployError(f"{model}:{version} has no serving_url; deploy a model server for it first")
        return {"model": model, "version": version, "url": mv["serving_url"],
                "price_per_1k_tokens": mv["metrics"].get("cost_per_1k_tokens", 0.0)}

    # ---------- writes ----------
    def create(self, name: str, model: str, version: str, actor: str = "system") -> dict[str, Any]:
        self._backend(model, version)
        with db.tx(self.conn) as c:
            if c.execute("SELECT 1 FROM deployments WHERE name=?", (name,)).fetchone():
                raise DeployError(f"deployment {name} exists")
            c.execute("INSERT INTO deployments(name, model, stable_version, status, updated_at) VALUES (?,?,?,?,?)",
                      (name, model, version, "stable", time.time()))
            self._event(c, name, "create", {"version": version}, actor, 1)
        self.registry.transition(model, version, "production", actor, f"initial deployment {name}")
        return self.get(name)

    def start_canary(self, name: str, version: str, weight: int = 10, actor: str = "system",
                     expected_revision: int | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        if not 0 < weight <= 100:
            raise DeployError("canary weight must be in (0, 100]")

        stage = self.registry.get(self.get(name)["model"], version)["stage"]
        if stage in ("rejected", "archived"):
            raise DeployError(f"{version} is {stage}; re-qualify it (transition to candidate) before a canary")

        def op(c, d):
            if d["stable_version"] == version:
                raise DeployError(f"{version} is already stable")
            if d["status"] == "canary" and d["canary_version"] != version:
                raise DeployError(f"canary {d['canary_version']} already in progress; promote or roll back first")
            self._backend(d["model"], version)
            c.execute("UPDATE deployments SET canary_version=?, canary_weight=?, status='canary' WHERE name=?",
                      (version, weight, name))
            return "start_canary", {"version": version, "weight": weight, "stable": d["stable_version"]}

        res = self._mutate(name, op, actor, expected_revision, idempotency_key)
        mv = self.registry.get(res["model"], version)
        if mv["stage"] == "candidate":
            self.registry.transition(res["model"], version, "staging", actor, f"canary on {name}")
        return res

    def set_weight(self, name: str, weight: int, actor: str = "system",
                   expected_revision: int | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        def op(c, d):
            if not d["canary_version"]:
                raise DeployError("no canary in progress")
            if not 0 <= weight <= 100:
                raise DeployError("weight must be in [0, 100]")
            c.execute("UPDATE deployments SET canary_weight=? WHERE name=?", (weight, name))
            return "set_weight", {"from": d["canary_weight"], "to": weight, "canary": d["canary_version"]}

        return self._mutate(name, op, actor, expected_revision, idempotency_key)

    def pause(self, name: str, actor: str = "system", idempotency_key: str | None = None) -> dict[str, Any]:
        def op(c, d):
            if d["status"] != "canary":
                raise DeployError("nothing to pause")
            c.execute("UPDATE deployments SET status='paused' WHERE name=?", (name,))
            return "pause", {"canary": d["canary_version"], "weight": d["canary_weight"]}

        return self._mutate(name, op, actor, None, idempotency_key)

    def promote(self, name: str, actor: str = "system", expected_revision: int | None = None,
                idempotency_key: str | None = None) -> dict[str, Any]:
        def op(c, d):
            if not d["canary_version"]:
                raise DeployError("no canary to promote")
            c.execute("UPDATE deployments SET stable_version=?, canary_version=NULL, canary_weight=0, status='stable'"
                      " WHERE name=?", (d["canary_version"], name))
            return "promote", {"from": d["stable_version"], "to": d["canary_version"]}

        res = self._mutate(name, op, actor, expected_revision, idempotency_key)
        self.registry.transition(res["model"], res["stable_version"], "production", actor, f"promoted on {name}")
        return res

    def rollback(self, name: str, actor: str = "system", reason: str = "", to_version: str | None = None,
                 expected_revision: int | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        """Abort an in-flight canary, or revert stable to the previous production version."""
        rolled: dict[str, Any] = {}

        def op(c, d):
            if d["canary_version"]:
                rolled.update(version=d["canary_version"], kind="canary")
                c.execute("UPDATE deployments SET canary_version=NULL, canary_weight=0, status='stable' WHERE name=?",
                          (name,))
                return "rollback", {"aborted_canary": d["canary_version"], "stable": d["stable_version"],
                                    "reason": reason}
            target = to_version or self._previous_stable(c, name, d["stable_version"])
            if not target:
                raise DeployError("no previous version to roll back to")
            self._backend(d["model"], target)
            rolled.update(version=d["stable_version"], kind="stable", target=target)
            c.execute("UPDATE deployments SET stable_version=?, status='stable' WHERE name=?", (target, name))
            return "rollback", {"from": d["stable_version"], "to": target, "reason": reason}

        res = self._mutate(name, op, actor, expected_revision, idempotency_key)
        if rolled.get("kind") == "canary":
            self.registry.transition(res["model"], rolled["version"], "rejected", actor, reason or "canary rolled back")
        elif rolled.get("kind") == "stable":
            self.registry.transition(res["model"], rolled["target"], "production", actor, reason or "rollback")
        return res

    def scale(self, name: str, replicas: int, actor: str = "system", idempotency_key: str | None = None) -> dict:
        if not 1 <= replicas <= 64:
            raise DeployError("replicas must be in [1, 64]")

        def op(c, d):
            c.execute("UPDATE deployments SET replicas=? WHERE name=?", (replicas, name))
            return "scale", {"from": d["replicas"], "to": replicas}

        return self._mutate(name, op, actor, None, idempotency_key)

    # ---------- internals ----------
    def _mutate(self, name, op, actor, expected_revision, idempotency_key) -> dict[str, Any]:
        if idempotency_key:
            hit = self.conn.execute("SELECT response FROM idempotency WHERE key=?", (idempotency_key,)).fetchone()
            if hit:
                return json.loads(hit["response"])
        with db.tx(self.conn) as c:
            d = db.row(c.execute("SELECT * FROM deployments WHERE name=?", (name,)).fetchone())
            if not d:
                raise DeployError(f"unknown deployment {name}")
            if expected_revision is not None and expected_revision != d["revision"]:
                raise DeployError(f"revision conflict: expected {expected_revision}, current {d['revision']}")
            action, detail = op(c, d)
            rev = d["revision"] + 1
            c.execute("UPDATE deployments SET revision=?, updated_at=? WHERE name=?", (rev, time.time(), name))
            self._event(c, name, action, detail, actor, rev)
            res = db.row(c.execute("SELECT * FROM deployments WHERE name=?", (name,)).fetchone())
            if idempotency_key:
                c.execute("INSERT INTO idempotency(key, response, at) VALUES (?,?,?)",
                          (idempotency_key, json.dumps(res), time.time()))
        return res

    @staticmethod
    def _event(c, name, action, detail, actor, revision):
        c.execute("INSERT INTO deployment_events(deployment, action, detail, actor, revision, at) VALUES (?,?,?,?,?,?)",
                  (name, action, json.dumps(detail), actor, revision, time.time()))

    @staticmethod
    def _previous_stable(c, name: str, current: str) -> str | None:
        for r in c.execute("SELECT action, detail FROM deployment_events WHERE deployment=? ORDER BY id DESC", (name,)):
            detail = json.loads(r["detail"])
            if r["action"] == "promote" and detail.get("to") == current:
                return detail.get("from")
            if r["action"] == "rollback" and detail.get("to") == current:
                continue
        return None

"""Model registry: versions, lineage, evaluation metrics and lifecycle stages.

Lifecycle:
    candidate -> staging -> production -> archived
                 \\-> rejected
Only one version per model may be in `production`; promoting a new one archives
the previous one atomically. MLflow, when configured, is a mirror, not the source
of truth: the platform owns lifecycle semantics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from forge import db

log = logging.getLogger(__name__)

STAGES = ("candidate", "staging", "production", "archived", "rejected")
ALLOWED = {
    "candidate": {"staging", "production", "rejected", "archived"},
    "staging": {"production", "rejected", "archived", "candidate"},
    "production": {"archived"},
    "archived": {"staging", "production"},  # rollback target
    "rejected": {"candidate"},
}


class RegistryError(Exception):
    pass


class Registry:
    def __init__(self, conn=None):
        self.conn = conn or db.connect()

    def register(
        self,
        name: str,
        version: str,
        base_model: str,
        adapter_uri: str | None = None,
        serving_url: str | None = None,
        dataset_hash: str | None = None,
        run_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        with db.tx(self.conn) as c:
            if c.execute("SELECT 1 FROM model_versions WHERE name=? AND version=?", (name, version)).fetchone():
                raise RegistryError(f"{name}:{version} already registered (versions are immutable)")
            c.execute(
                "INSERT INTO model_versions(name, version, base_model, adapter_uri, serving_url, dataset_hash,"
                " run_id, stage, metrics, tags, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (name, version, base_model, adapter_uri, serving_url, dataset_hash, run_id, "candidate",
                 json.dumps(metrics or {}), json.dumps(tags or {}), time.time()),
            )
            c.execute(
                "INSERT INTO stage_transitions(name, version, from_stage, to_stage, actor, reason, at)"
                " VALUES (?,?,?,?,?,?,?)",
                (name, version, None, "candidate", actor, "registered", time.time()),
            )
        self._mirror_mlflow(name, version, metrics or {})
        return self.get(name, version)

    def get(self, name: str, version: str) -> dict[str, Any]:
        r = db.row(
            self.conn.execute("SELECT * FROM model_versions WHERE name=? AND version=?", (name, version)).fetchone(),
            ("metrics", "tags"),
        )
        if not r:
            raise RegistryError(f"unknown model version {name}:{version}")
        return r

    def list(self, name: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM model_versions", ()
        if name:
            q, args = q + " WHERE name=?", (name,)
        return [db.row(r, ("metrics", "tags")) for r in self.conn.execute(q + " ORDER BY name, created_at", args)]

    def production(self, name: str) -> dict[str, Any] | None:
        return db.row(
            self.conn.execute(
                "SELECT * FROM model_versions WHERE name=? AND stage='production'", (name,)
            ).fetchone(),
            ("metrics", "tags"),
        )

    def update_metrics(self, name: str, version: str, metrics: dict[str, Any]) -> dict[str, Any]:
        current = self.get(name, version)["metrics"]
        current.update(metrics)
        with db.tx(self.conn) as c:
            c.execute("UPDATE model_versions SET metrics=? WHERE name=? AND version=?",
                      (json.dumps(current), name, version))
        self._mirror_mlflow(name, version, metrics)
        return self.get(name, version)

    def set_serving_url(self, name: str, version: str, url: str) -> None:
        self.get(name, version)
        with db.tx(self.conn) as c:
            c.execute("UPDATE model_versions SET serving_url=? WHERE name=? AND version=?", (url, name, version))

    def transition(self, name: str, version: str, to_stage: str, actor: str = "system", reason: str = "") -> dict:
        if to_stage not in STAGES:
            raise RegistryError(f"unknown stage {to_stage}")
        with db.tx(self.conn) as c:
            cur = c.execute("SELECT stage FROM model_versions WHERE name=? AND version=?", (name, version)).fetchone()
            if not cur:
                raise RegistryError(f"unknown model version {name}:{version}")
            frm = cur["stage"]
            if frm == to_stage:
                return self._get_in(c, name, version)
            if to_stage not in ALLOWED[frm]:
                raise RegistryError(f"illegal transition {frm} -> {to_stage} for {name}:{version}")
            if to_stage == "production":
                for prev in c.execute(
                    "SELECT version FROM model_versions WHERE name=? AND stage='production'", (name,)
                ).fetchall():
                    c.execute("UPDATE model_versions SET stage='archived' WHERE name=? AND version=?",
                              (name, prev["version"]))
                    c.execute(
                        "INSERT INTO stage_transitions(name, version, from_stage, to_stage, actor, reason, at)"
                        " VALUES (?,?,?,?,?,?,?)",
                        (name, prev["version"], "production", "archived", actor, f"superseded by {version}",
                         time.time()),
                    )
            c.execute("UPDATE model_versions SET stage=? WHERE name=? AND version=?", (to_stage, name, version))
            c.execute(
                "INSERT INTO stage_transitions(name, version, from_stage, to_stage, actor, reason, at)"
                " VALUES (?,?,?,?,?,?,?)",
                (name, version, frm, to_stage, actor, reason, time.time()),
            )
            return self._get_in(c, name, version)

    def history(self, name: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM stage_transitions WHERE name=? ORDER BY id", (name,))]

    @staticmethod
    def _get_in(c, name, version):
        return db.row(c.execute("SELECT * FROM model_versions WHERE name=? AND version=?", (name, version)).fetchone(),
                      ("metrics", "tags"))

    @staticmethod
    def _mirror_mlflow(name: str, version: str, metrics: dict[str, Any]) -> None:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return
        try:
            import mlflow

            with mlflow.start_run(run_name=f"{name}:{version}"):
                mlflow.set_tags({"forge.model": name, "forge.version": version})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception as e:  # mirror failures must never break the registry
            log.warning("mlflow mirror failed: %s", e)

"""SQLite state store shared by the registry, deploy controller, incidents and audit log.

SQLite in WAL mode is plenty for a single control-plane replica. The schema is
deliberately plain SQL so swapping to Postgres is a connection-string change.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from forge.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    base_model TEXT NOT NULL,
    adapter_uri TEXT,
    serving_url TEXT,
    dataset_hash TEXT,
    run_id TEXT,
    stage TEXT NOT NULL DEFAULT 'candidate',
    metrics TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE TABLE IF NOT EXISTS stage_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, version TEXT, from_stage TEXT, to_stage TEXT, actor TEXT, reason TEXT, at REAL
);
CREATE TABLE IF NOT EXISTS deployments (
    name TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    stable_version TEXT,
    canary_version TEXT,
    canary_weight INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'stable',
    revision INTEGER NOT NULL DEFAULT 1,
    replicas INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deployment_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment TEXT, action TEXT, detail TEXT, actor TEXT, revision INTEGER, at REAL
);
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    title TEXT, severity TEXT, state TEXT, source TEXT,
    labels TEXT, diagnosis TEXT, proposed_action TEXT, timeline TEXT,
    created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY, response TEXT, at REAL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT, action TEXT, target TEXT, detail TEXT, at REAL
);
"""

_lock = threading.Lock()
_conns: dict[str, sqlite3.Connection] = {}


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = str(path or settings().db_path)
    with _lock:
        if p not in _conns:
            conn = sqlite3.connect(p, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(SCHEMA)
            _conns[p] = conn
        return _conns[p]


@contextmanager
def tx(conn: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Serialized write transaction (BEGIN IMMEDIATE takes the write lock up front)."""
    c = conn or connect()
    with _lock:
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise


def row(r: sqlite3.Row | None, json_cols: tuple[str, ...] = ()) -> dict[str, Any] | None:
    if r is None:
        return None
    d = dict(r)
    for c in json_cols:
        if d.get(c):
            d[c] = json.loads(d[c])
    return d


def audit(actor: str, action: str, target: str, detail: Any = None) -> None:
    with tx() as c:
        c.execute(
            "INSERT INTO audit(actor, action, target, detail, at) VALUES (?,?,?,?,?)",
            (actor, action, target, json.dumps(detail, default=str), time.time()),
        )

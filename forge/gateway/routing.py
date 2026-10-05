"""Routing table: converges on the control plane's desired state.

The gateway polls the control plane and keeps the last-known-good table if the
control plane is unreachable. The data plane must keep serving when the control plane
is down (graceful degradation). Staleness is exported as a metric so it can be alerted on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass

import httpx

from forge.gateway import metrics as m

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Backend:
    model: str
    version: str
    url: str
    weight: int
    role: str
    price_per_1k_tokens: float

    @property
    def key(self) -> str:
        return f"{self.model}:{self.version}"

    @property
    def served_model(self) -> str:
        return f"{self.model}-{self.version}"


@dataclass
class Route:
    revision: int
    status: str
    backends: list[Backend]
    fallback: Backend | None


class RoutingTable:
    def __init__(self, controlplane_url: str, api_key: str, refresh_s: float, static_file: str | None = None):
        self.url = controlplane_url.rstrip("/")
        self.api_key = api_key
        self.refresh_s = refresh_s
        self.static_file = static_file
        self.routes: dict[str, Route] = {}
        self.last_ok = 0.0
        self._task: asyncio.Task | None = None

    @staticmethod
    def parse(raw: dict) -> dict[str, Route]:
        out = {}
        for name, r in raw.items():
            backends = [Backend(b["model"], b["version"], b["url"], b["weight"], b["role"],
                                b.get("price_per_1k_tokens", 0.0)) for b in r["backends"]]
            fb = r.get("fallback")
            fallback = Backend(fb["model"], fb["version"], fb["url"], 0, "fallback",
                               fb.get("price_per_1k_tokens", 0.0)) if fb else None
            out[name] = Route(r["revision"], r["status"], backends, fallback)
        return out

    def apply(self, raw: dict) -> None:
        new = self.parse(raw)
        for name, r in new.items():
            old = self.routes.get(name)
            if not old or old.revision != r.revision:
                log.info("routing %s -> revision %s: %s", name, r.revision,
                         [(b.version, b.role, b.weight) for b in r.backends])
            m.ROUTE_REV.labels(name).set(r.revision)
            for b in r.backends:
                m.CANARY_WEIGHT.labels(name, b.version).set(b.weight if b.role == "canary" else 0)
        self.routes = new
        self.last_ok = time.time()

    async def refresh_once(self, client: httpx.AsyncClient) -> None:
        if self.static_file and not self.url:
            self.apply(json.loads(open(self.static_file).read()))
            return
        r = await client.get(f"{self.url}/api/routing", headers={"Authorization": f"Bearer {self.api_key}"},
                             timeout=2.0)
        r.raise_for_status()
        self.apply(r.json())

    async def run(self) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    await self.refresh_once(client)
                except Exception as e:
                    log.warning("routing refresh failed, serving last-known-good table: %s", e)
                    if not self.routes and self.static_file and os.path.exists(self.static_file):
                        self.apply(json.loads(open(self.static_file).read()))
                m.ROUTING_STALE.set(time.time() - self.last_ok if self.last_ok else -1)
                await asyncio.sleep(self.refresh_s)

    def start(self) -> None:
        self._task = asyncio.create_task(self.run())

    def plan(self, deployment: str, sticky_key: str | None, attempts: int) -> list[Backend]:
        """Ordered list of backends to try: weighted primary, then alternates, then primary again."""
        route = self.routes.get(deployment)
        if not route or not route.backends:
            return []
        rnd = random.Random(sticky_key) if sticky_key else random
        total = sum(b.weight for b in route.backends)
        pick = rnd.uniform(0, total) if total else 0
        primary = route.backends[0]
        acc = 0.0
        for b in route.backends:
            acc += b.weight
            if pick <= acc and b.weight > 0:
                primary = b
                break
        alternates = [b for b in route.backends if b is not primary]
        if route.fallback and all(a.version != route.fallback.version for a in alternates) \
                and route.fallback.version != primary.version:
            alternates.append(route.fallback)
        order = [primary, *alternates, primary]
        return order[: max(1, attempts)]

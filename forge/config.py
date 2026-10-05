"""Platform configuration, read once from the environment.

Every service (gateway, control plane, simulator, agent) shares this so a single
`.env` / Helm values file configures the whole platform.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_api_keys(raw: str) -> dict[str, tuple[str, str]]:
    """`key:tenant:role,key2:tenant2:role2` -> {key: (tenant, role)}."""
    out: dict[str, tuple[str, str]] = {}
    for item in filter(None, (s.strip() for s in raw.split(","))):
        key, tenant, role = item.split(":")
        out[key] = (tenant, role)
    return out


def _parse_tenant_limits(raw: str) -> dict[str, tuple[float, int, int]]:
    """`tenant=rps/burst/priority,...` -> {tenant: (rps, burst, priority)}. Lower priority is shed first."""
    out: dict[str, tuple[float, int, int]] = {}
    for item in filter(None, (s.strip() for s in raw.split(","))):
        tenant, spec = item.split("=")
        rps, burst, prio = spec.split("/")
        out[tenant] = (float(rps), int(burst), int(prio))
    return out


@dataclass(frozen=True)
class Settings:
    home: Path = field(default_factory=lambda: Path(_env("FORGE_HOME", ".forge")).resolve())
    controlplane_url: str = field(default_factory=lambda: _env("FORGE_CONTROLPLANE_URL", "http://localhost:8080"))
    gateway_url: str = field(default_factory=lambda: _env("FORGE_GATEWAY_URL", "http://localhost:8000"))
    prometheus_url: str = field(default_factory=lambda: _env("FORGE_PROMETHEUS_URL", "http://localhost:9090"))
    tempo_url: str = field(default_factory=lambda: _env("FORGE_TEMPO_URL", "http://localhost:3200"))
    otlp_endpoint: str = field(default_factory=lambda: _env("OTEL_EXPORTER_OTLP_ENDPOINT", ""))

    # Security
    api_keys: dict[str, tuple[str, str]] = field(
        default_factory=lambda: _parse_api_keys(
            _env("FORGE_API_KEYS", "dev-admin:platform:admin,dev-acme:acme:operator,dev-free:free-tier:viewer")
        )
    )
    tenant_limits: dict[str, tuple[float, int, int]] = field(
        default_factory=lambda: _parse_tenant_limits(
            _env("FORGE_TENANT_LIMITS", "platform=100/200/2,acme=20/40/1,free-tier=5/10/0")
        )
    )
    max_prompt_chars: int = field(default_factory=lambda: int(_env("FORGE_MAX_PROMPT_CHARS", "16000")))

    # Gateway reliability knobs
    max_inflight_per_backend: int = field(default_factory=lambda: int(_env("FORGE_MAX_INFLIGHT", "32")))
    max_queue: int = field(default_factory=lambda: int(_env("FORGE_MAX_QUEUE", "64")))
    queue_timeout_s: float = field(default_factory=lambda: float(_env("FORGE_QUEUE_TIMEOUT_S", "5")))
    upstream_timeout_s: float = field(default_factory=lambda: float(_env("FORGE_UPSTREAM_TIMEOUT_S", "30")))
    ttft_timeout_s: float = field(default_factory=lambda: float(_env("FORGE_TTFT_TIMEOUT_S", "10")))
    retries: int = field(default_factory=lambda: int(_env("FORGE_RETRIES", "2")))
    breaker_failure_threshold: float = field(default_factory=lambda: float(_env("FORGE_BREAKER_THRESHOLD", "0.5")))
    breaker_min_calls: int = field(default_factory=lambda: int(_env("FORGE_BREAKER_MIN_CALLS", "10")))
    breaker_open_s: float = field(default_factory=lambda: float(_env("FORGE_BREAKER_OPEN_S", "15")))
    routing_refresh_s: float = field(default_factory=lambda: float(_env("FORGE_ROUTING_REFRESH_S", "2")))

    # SLOs the agent and alerts are built around
    slo_ttft_p95_s: float = field(default_factory=lambda: float(_env("FORGE_SLO_TTFT_P95_S", "0.5")))
    slo_latency_p95_s: float = field(default_factory=lambda: float(_env("FORGE_SLO_LATENCY_P95_S", "3.0")))
    slo_error_rate: float = field(default_factory=lambda: float(_env("FORGE_SLO_ERROR_RATE", "0.02")))

    # Agent
    agent_model: str = field(default_factory=lambda: _env("FORGE_AGENT_MODEL", "claude-opus-5-5"))
    agent_llm: str = field(default_factory=lambda: _env("FORGE_AGENT_LLM", "auto"))  # auto | claude | heuristic
    verify_window_s: float = field(default_factory=lambda: float(_env("FORGE_VERIFY_WINDOW_S", "60")))

    @property
    def db_path(self) -> Path:
        return self.home / "forge.db"

    @property
    def artifacts(self) -> Path:
        return self.home / "artifacts"


@lru_cache
def settings() -> Settings:
    s = Settings()
    s.home.mkdir(parents=True, exist_ok=True)
    s.artifacts.mkdir(parents=True, exist_ok=True)
    return s

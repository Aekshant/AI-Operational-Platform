"""Gateway metrics. These are the platform's user-facing SLIs."""

from prometheus_client import Counter, Gauge, Histogram

DV = ["deployment", "version"]
LAT = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 20, 30)

REQUESTS = Counter("forge_requests_total", "Inference requests", DV + ["tenant", "code"])
DURATION = Histogram("forge_request_duration_seconds", "End-to-end request latency", DV, buckets=LAT)
TTFT = Histogram("forge_ttft_seconds", "Time to first token", DV, buckets=LAT)
TPOT = Histogram("forge_tpot_seconds", "Time per output token", DV,
                 buckets=(0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2))
TOKENS = Counter("forge_tokens_total", "Tokens", DV + ["tenant", "kind"])
COST = Counter("forge_cost_usd_total", "Attributed serving cost (USD)", DV + ["tenant"])
INFLIGHT = Gauge("forge_inflight_requests", "In-flight requests per backend", ["backend"])
QUEUE = Gauge("forge_queue_depth", "Requests waiting in admission queue per backend", ["backend"])
REJECTED = Counter("forge_rejected_total", "Rejected requests", ["reason", "tenant"])
FALLBACK = Counter("forge_fallback_total", "Requests served by fallback", ["deployment", "from_version", "reason"])
RETRIES = Counter("forge_retries_total", "Upstream retries", ["backend", "kind"])
UPSTREAM_ERR = Counter("forge_upstream_errors_total", "Upstream failures", ["backend", "kind"])
BREAKER = Gauge("forge_breaker_state", "Circuit breaker state (0 closed, 1 half-open, 2 open)", ["backend"])
ROUTE_REV = Gauge("forge_routing_revision", "Routing table revision applied", ["deployment"])
CANARY_WEIGHT = Gauge("forge_canary_weight", "Canary traffic weight applied", ["deployment", "version"])
ROUTING_STALE = Gauge("forge_routing_stale_seconds", "Seconds since last successful routing refresh")

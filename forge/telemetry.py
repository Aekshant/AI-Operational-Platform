"""OpenTelemetry bootstrap: traces (-> Tempo) and logs (-> Loki) over OTLP.

Metrics stay on the Prometheus client (pull model, `/metrics`) because that's what
vLLM, DCGM and kube-state-metrics speak, so every layer lands in one Prometheus.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from forge.config import settings

_configured = False


def setup(service_name: str, app=None) -> trace.Tracer:
    global _configured
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per upstream call drowns real signal
    if not _configured:
        resource = Resource.create({"service.name": service_name, "service.namespace": "forgeai"})
        provider = TracerProvider(resource=resource)
        endpoint = settings().otlp_endpoint
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")))
            _setup_logs(resource, endpoint)
        trace.set_tracer_provider(provider)
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
        except Exception:  # instrumentation is best-effort; never block startup on it
            logging.getLogger(__name__).warning("httpx instrumentation unavailable")
        _configured = True
    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,healthz")
    return trace.get_tracer(service_name)


def _setup_logs(resource: Resource, endpoint: str) -> None:
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        lp = LoggerProvider(resource=resource)
        lp.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{endpoint}/v1/logs")))
        set_logger_provider(lp)
        logging.getLogger().addHandler(LoggingHandler(level=logging.INFO, logger_provider=lp))
    except Exception:
        logging.getLogger(__name__).warning("OTLP log export unavailable; logs stay on stdout")


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None

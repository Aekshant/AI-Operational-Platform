FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FORGE_HOME=/var/lib/forge
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
COPY pyproject.toml README.md ./
COPY forge ./forge
RUN uv pip install --system . && useradd -r -u 10001 forge && mkdir -p /var/lib/forge && chown forge /var/lib/forge
COPY data ./data
COPY scripts ./scripts
USER forge
EXPOSE 8000 8080
CMD ["forge", "--help"]

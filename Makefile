COMPOSE = docker compose -f deploy/docker-compose.yml
EXEC = $(COMPOSE) exec -T controlplane

.PHONY: install test up down demo load load-stop incident approve cost status logs dashboard

install:            ## local dev environment (Python 3.11 via uv)
	uv venv -p 3.11 .venv && uv pip install -p .venv -e ".[dev]"

test:
	.venv/bin/pytest -q

up:                 ## build and start the whole platform
	$(COMPOSE) up -d --build
	@echo "UI http://localhost:8080  ·  Grafana http://localhost:3000  ·  Prometheus http://localhost:9090"

down:
	$(COMPOSE) --profile load down -v

demo:               ## register + evaluate v1..v3, deploy v1, ship v2
	$(EXEC) forge demo setup

load:               ## start background traffic
	$(COMPOSE) --profile load up -d loadgen

load-stop:
	$(COMPOSE) stop loadgen

incident:           ## canary v3 → SLO alert → agent → approval
	$(EXEC) forge demo incident

approve:            ## approve the open incident: make approve ID=inc-...
	$(EXEC) forge incident approve $(ID)

cost:               ## output-token regression → "why did cost go up?"
	$(EXEC) forge demo cost

status:
	$(EXEC) forge platform status

logs:
	$(COMPOSE) logs -f --tail=50 controlplane gateway

dashboard:
	.venv/bin/python scripts/gen_dashboard.py

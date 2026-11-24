.PHONY: install test lint run up audit-verify
install:
	pip install -e ".[dev]"
test:
	python -m pytest -q
lint:
	ruff check . && ruff format --check .
run:
	uvicorn ent_agent.api:create_app --factory --reload --port 8000
up:
	docker compose up --build
audit-verify:
	python -c "from ent_agent.audit import AuditLog; print(AuditLog.verify('audit.log.jsonl'))"

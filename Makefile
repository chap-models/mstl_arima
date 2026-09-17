.PHONY: help run build run-ghcr test test-docker lint check parity clean

IMAGE      ?= chap-mstl-arima:latest
GHCR_IMAGE ?= ghcr.io/chap-models/mstl_arima:latest
TEST_NAME  ?= chap-mstl-arima-test
TEST_PORT  ?= 9000
TEST_URL   ?= http://localhost:$(TEST_PORT)
PARITY_URL ?= http://localhost:9090

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run          Run the service locally on :9090 (uv run python main.py)"
	@echo "  build        Build the docker image ($(IMAGE))"
	@echo "  run-ghcr     Run the prebuilt GHCR image ($(GHCR_IMAGE)) on :9090"
	@echo "  test         pytest, in-process via FastAPI TestClient (includes golden parity)"
	@echo "  test-docker  Build, start the container, run 'chapkit test' against it, shut down"
	@echo "  parity       Compare a running service at $(PARITY_URL) with the legacy goldens"
	@echo "  lint         ruff format + ruff check --fix (modifies files)"
	@echo "  check        ruff format --check + ruff check (CI, does not modify files)"
	@echo "  clean        Remove caches, build artifacts and the local SQLite database"

run:
	@uv run python main.py

build:
	@echo ">>> Building $(IMAGE)"
	@docker build --build-arg GIT_REVISION=$(shell git rev-parse HEAD 2>/dev/null) -t $(IMAGE) .

run-ghcr:
	@echo ">>> Running $(GHCR_IMAGE) on :9090"
	@docker run --rm --pull always -p 9090:8000 --name chap-mstl-arima $(GHCR_IMAGE)

test:
	@uv run pytest -v

test-docker: build
	@echo ">>> Starting $(TEST_NAME) on :$(TEST_PORT)"
	@docker rm -f $(TEST_NAME) >/dev/null 2>&1 || true
	@docker run -d --rm -p $(TEST_PORT):8000 --name $(TEST_NAME) $(IMAGE) >/dev/null
	@trap 'echo ">>> Stopping $(TEST_NAME)"; docker stop $(TEST_NAME) >/dev/null 2>&1 || true' EXIT; \
		echo ">>> Waiting for $(TEST_URL)/health"; \
		for i in $$(seq 1 60); do \
			if curl -fsS $(TEST_URL)/health >/dev/null 2>&1; then break; fi; \
			if [ $$i -eq 60 ]; then \
				echo ">>> Service did not become healthy in time"; \
				docker logs $(TEST_NAME) || true; \
				exit 1; \
			fi; \
			sleep 1; \
		done; \
		echo ">>> Running chapkit test against $(TEST_URL)"; \
		uv run chapkit test --url $(TEST_URL) --timeout 300; \
		echo ">>> Running chapkit test (weekly) against $(TEST_URL)"; \
		uv run chapkit test --url $(TEST_URL) --period-type weekly --rows 520 --predict-rows 300 --timeout 300

parity:
	@uv run python scripts/parity.py --url $(PARITY_URL) --kind monthly
	@uv run python scripts/parity.py --url $(PARITY_URL) --kind weekly

lint:
	@uv run ruff format .
	@uv run ruff check --fix .

check:
	@uv run ruff format --check .
	@uv run ruff check .

clean:
	@rm -rf data target dist build .pytest_cache .ruff_cache .mypy_cache *.egg-info
	@find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -not -path "./.venv/*" -delete
	@rm -f model.json predictions.csv
	@echo "cleaned"

.DEFAULT_GOAL := help

# QSMPC-QKD-QHE-AI-Hybrid — common tasks
# Usage: `make <target>`. Python 3.12 venv lives in .venv/

PYTHON ?= python3.12
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

.DEFAULT_GOAL := help
.PHONY: help venv setup setup-ai setup-fhe setup-api submodules data test lint \
        run-finance run-medical benchmark api web docker-up docker-demo clean check-secrets \
        ci-local

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the python3.12 virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel setuptools

setup: venv ## Install core (crypto+quantum CPU) deps + package (editable)
	$(PIP) install -r requirements-core.txt
	$(PIP) install -e .

setup-ai: ## Install AI + federated-learning deps
	$(PIP) install -r requirements-ai.txt

setup-fhe: ## Install FHE backends (TenSEAL + OpenFHE, both commercial-OK; see file)
	$(PIP) install -r requirements-fhe.txt

setup-api: ## Install API (FastAPI) deps
	$(PIP) install -r requirements-api.txt

submodules: ## Init/update git submodules (upstream OSS)
	bash scripts/bootstrap_submodules.sh

data: ## Download free public datasets + generate synthetic data
	$(PY) -m qsh.data.download

test: ## Run the test suite (pytest prints the count; -m "not slow" deselects the gated tests)
	# The help text above used to quote "174 collected, 170 passed". It was 236 by the time
	# anyone noticed, and `make help` prints it, so a newcomer's first number was wrong by a
	# third. A count that has to be hand-maintained in a comment will drift; pytest reports it
	# on every run, which is the version that cannot go stale.
	#
	# -m "not slow" deselects the Node- and OpenFHE-gated tests. Without it the OpenFHE
	# bootstrap test SIGSEGVs inside EvalBootstrapSetup on AlmaLinux 9.7 -- the import guard
	# in fhe/backends.py catches an ImportError, not a segfault, so the process just dies.
	$(PY) -m pytest -m "not slow"

test-all: ## Run every test including slow ones (may segfault on OpenFHE, see above)
	$(PY) -m pytest

check-pqc: ## FIPS 203 s7.2 negative vectors + FIPS 204 sizes, vs the vendored CC0 snapshot
	# Offline: reads web/scripts/vectors. ARGS=--live checks the current published set instead.
	# Also runs as gate 8 of ci-local and in CI; this target is for running it on its own.
	cd web && npm run --silent check:pqc -- $(ARGS)

lint: ## Ruff lint
	$(VENV)/bin/ruff check src tests

run-finance: ## Run the Finance AML/PSI E2E demo
	$(PY) -m qsh.orchestration.pipeline --usecase finance_aml --out results/

run-medical: ## Run the Medical federated-learning E2E demo
	$(PY) -m qsh.orchestration.pipeline --usecase medical_fl --out results/

benchmark: ## Run MPC-FHE vs QSMPC-FHE benchmark
	bash scripts/run_benchmark.sh

api: ## Start the FastAPI backend (dev)
	$(VENV)/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

web: ## Start the SPA dev server
	cd web && npm install && npm run dev

docker-up: ## Local full-stack via docker compose
	docker compose -f docker/docker-compose.yml up --build

docker-demo: ## Lightweight public-demo stack (VPS profile)
	docker compose -f docker/docker-compose.demo.yml up --build

check-secrets: ## Assert the report .docx and monetization notes are NOT tracked
	@! git ls-files | grep -qi 'MONETIZATION' && echo "OK: MONETIZATION notes untracked" || (echo "FAIL: a MONETIZATION file is tracked!" && exit 1)
	@! git ls-files | grep -q '\.docx$$' && echo "OK: report .docx untracked" || (echo "FAIL: a .docx is tracked!" && exit 1)

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

ci-local: ## Run every BLOCKING gate .github/workflows/ci.yml runs, in CI's own order
	@bash scripts/ci_local.sh

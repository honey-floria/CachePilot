.PHONY: install-cpu lint test check phase0-exit serve serve-empty colab-acceptance ops-export

PYTHON ?= python3
BASE_URL ?= http://127.0.0.1:8000
RUN_DIR ?= runs/demo-sim-mixed-001
OUTPUT_DIR ?= artifacts/diagnostics

install-cpu:
	$(PYTHON) -m pip install -r requirements/requirements-cpu.txt

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

check: lint test

phase0-exit:
	$(PYTHON) benchmarks/phase0_exit.py

serve:
	$(PYTHON) -m cachepilot.gateway.api

serve-empty:
	$(PYTHON) -m cachepilot.runtime.empty_service

colab-acceptance:
	$(PYTHON) benchmarks/colab_acceptance.py

ops-export:
	$(PYTHON) benchmarks/export_diagnostics.py --base-url $(BASE_URL) --run-dir $(RUN_DIR) --output-dir $(OUTPUT_DIR)

.PHONY: install-cpu lint test check phase0-exit serve serve-empty colab-acceptance

PYTHON ?= python3

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

.PHONY: install-cpu lint test check serve colab-acceptance

PYTHON ?= python3

install-cpu:
	$(PYTHON) -m pip install -r requirements/requirements-cpu.txt

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

check: lint test

serve:
	$(PYTHON) -m cachepilot.runtime.empty_service

colab-acceptance:
	$(PYTHON) benchmarks/colab_acceptance.py

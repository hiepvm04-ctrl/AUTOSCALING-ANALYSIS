.PHONY: help install install-dev format lint type test clean preprocess features train benchmark simulate ui all

PY ?= python
CONFIG ?= configs/config.yaml

help:
	@echo "Targets:"
	@echo "  install        - pip install -e ."
	@echo "  install-dev    - pip install -e .[dev]"
	@echo "  format         - black + ruff --fix"
	@echo "  lint           - ruff"
	@echo "  type           - mypy"
	@echo "  test           - pytest"
	@echo "  preprocess     - raw logs -> ts3"
	@echo "  features       - ts3 -> features"
	@echo "  train          - train models + preds + metrics"
	@echo "  benchmark      - build benchmark table"
	@echo "  simulate       - run autoscaling sim (default hits/5m/xgb)"
	@echo "  ui             - run streamlit"
	@echo "  all            - preprocess + features + train + benchmark + simulate"
	@echo "  clean          - remove generated outputs"

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

format:
	black src scripts tests
	ruff check --fix src scripts tests

lint:
	ruff check src scripts tests

type:
	mypy src/autoscaling_analysis

test:
	pytest

preprocess:
	$(PY) scripts/preprocess.py --config $(CONFIG)

features:
	$(PY) scripts/features.py --config $(CONFIG)

train:
	$(PY) scripts/train.py --config $(CONFIG)

benchmark:
	$(PY) scripts/benchmark.py --config $(CONFIG)

simulate:
	$(PY) scripts/simulate_scaling.py --config $(CONFIG) --metric hits --window 5m --model xgb

ui:
	bash scripts/run_ui.sh $(CONFIG)

all: preprocess features train benchmark simulate

clean:
	rm -rf data/interim/* data/processed/*
	rm -rf artifacts/models/* artifacts/predictions/* artifacts/metrics/* artifacts/scaling/*
	rm -rf reports/eda/* reports/figures/*
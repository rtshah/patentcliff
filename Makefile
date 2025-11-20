.PHONY: help data build_labels features train serve test clean

help:
	@echo "Available targets:"
	@echo "  make data          - Pull sample data from APIs and cache"
	@echo "  make build_labels  - Build labels.parquet from events"
	@echo "  make features      - Build features.parquet"
	@echo "  make train         - Train models and log to MLflow"
	@echo "  make serve         - Run FastAPI server on :8000"
	@echo "  make test          - Run pytest suite"
	@echo "  make clean         - Remove artifacts and cache"

data:
	python -m src.connectors.nadac_client --discover-registry
	python -m src.connectors.openfda_client --sample
	python -m src.connectors.partd_client --sample

build_labels:
	python -m src.label.build_events
	python -m src.label.build_label

features:
	python -m src.model.dataset

train:
	python -m src.model.train

serve:
	uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=src

clean:
	rm -rf artifacts/*.parquet
	rm -rf config/cache/*
	rm -rf mlruns/
	rm -rf reports/*.png reports/*.json
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete


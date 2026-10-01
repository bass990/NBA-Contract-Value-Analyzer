.PHONY: install install-api scrape features train app api test lint docker-build docker-run docker-smoke notebook clean help

PYTHON ?= python

help:
	@echo "NBA contract value: make targets"
	@echo "  install        full training/dashboard environment (requirements.txt)"
	@echo "  install-api    lean inference environment (requirements-api.txt + pytest, httpx, ruff)"
	@echo "  train          deterministic synthetic run -> data/processed/model.pkl + artifacts/eval_report.json"
	@echo "  predictions    score the held-out season with the committed bundle -> data/processed/predictions_latest.csv"
	@echo "  scrape         Basketball-Reference stats for 2022-2025 (real-data path; cached to data/raw/_cache)"
	@echo "  app            Streamlit dashboard"
	@echo "  api            uvicorn api.main:app --reload --port 8000"
	@echo "  test           pytest tests/ (feature pipeline, API contract, golden, determinism, drift)"
	@echo "  lint           ruff check src tests api scripts"
	@echo "  docker-smoke   build the inference image, run it, curl /health /model-info /predict, stop"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-api:
	$(PYTHON) -m pip install -r requirements-api.txt pytest httpx ruff

train:
	$(PYTHON) scripts/train.py

predictions:
	$(PYTHON) scripts/predict_latest.py

scrape:
	$(PYTHON) -c "from src.scraper.basketball_reference import scrape_seasons; from pathlib import Path; scrape_seasons([2022,2023,2024,2025], Path('data/raw/_cache'), Path('data/raw'))"

app:
	$(PYTHON) -m streamlit run src/app/dashboard.py

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check src tests api scripts

docker-build:
	docker build -t nba-api .

docker-run: docker-build
	docker run --rm -p 8000:8000 nba-api

docker-smoke: docker-build
	docker run -d --name nba-smoke -p 8000:8000 nba-api
	sleep 8
	curl -sf http://localhost:8000/health
	curl -sf http://localhost:8000/model-info | head -c 300
	curl -sf -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @tests/example_request.json
	docker rm -f nba-smoke

notebook:
	$(PYTHON) -m jupyter notebook notebooks/NBA_Contract_Value_END_TO_END.ipynb

clean:
	rm -rf data/raw/_cache/*.html .pytest_cache .ruff_cache
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

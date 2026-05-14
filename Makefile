.PHONY: install scrape features train app notebook test clean all

# Default: full pipeline
all: install scrape features train

install:
	pip install -r requirements.txt

scrape:
	python -c "from src.scraper.basketball_reference import scrape_seasons; from pathlib import Path; scrape_seasons([2022,2023,2024,2025], Path('data/raw/_cache'), Path('data/raw'))"

features:
	python -c "from src.pipeline.synthetic import generate_synthetic_dataset; from src.pipeline.features import build_feature_table; import pandas as pd; pg,adv,sal=generate_synthetic_dataset(); ft=build_feature_table(pg,adv,sal); ft.to_csv('data/processed/features.csv', index=False); print('Features written.')"

train:
	python -c "from src.pipeline.synthetic import generate_synthetic_dataset; from src.pipeline.features import build_feature_table; from src.model.train import train, save_model; from pathlib import Path; pg,adv,sal=generate_synthetic_dataset(); ft=build_feature_table(pg,adv,sal); r=train(ft, test_season=2025, skip_tuning=True); save_model(r, Path('data/processed/model.pkl')); print(f'Test R^2: {r.test_r2:.3f}')"

app:
	streamlit run src/app/dashboard.py

notebook:
	jupyter notebook notebooks/NBA_Contract_Value_END_TO_END.ipynb

test:
	pytest tests/ -v

clean:
	rm -rf data/raw/_cache/*.html
	rm -rf data/processed/*.csv data/processed/*.pkl
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

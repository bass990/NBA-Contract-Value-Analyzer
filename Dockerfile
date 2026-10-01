# Lean inference image for the NBA contract-value API.
# Build:  docker build -t nba-api .
# Run:    docker run --rm -p 8000:8000 nba-api      # website at /, API docs at /docs
# Health: curl http://localhost:8000/health

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is required by LightGBM's compiled wheels at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt

# The API plus the static website it serves at / (no scrapers, no notebooks).
COPY src/ ./src/
COPY api/ ./api/
COPY website/ ./website/
COPY data/processed/model.pkl ./data/processed/model.pkl
COPY artifacts/eval_report.json ./artifacts/eval_report.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

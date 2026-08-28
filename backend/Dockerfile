FROM python:3.12-slim

# OpenCV needs a couple of system libs even in "headless" mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV APP_DATA_DIR=/app/data
RUN mkdir -p /app/data/uploads

# Train the model at build time so the image is self-contained and
# deterministic (fixed random seed in app/ml/train.py). If artifacts already
# exist (e.g. copied in), training is skipped.
RUN python -c "from app.ml.infer import QualityModel; import sys; sys.exit(0 if QualityModel.artifacts_available() else 1)" \
    || python -m app.ml.train

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

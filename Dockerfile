FROM python:3.11-slim

# ocrmypdf + tesseract (eng+hin) for pipeline/extract.py's OCR fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ocrmypdf tesseract-ocr tesseract-ocr-hin poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_trf

COPY . .

# Default: run the web app. Pipeline stages are run with
# `docker compose run --rm pipeline python -m pipeline.run <stage> ...`
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

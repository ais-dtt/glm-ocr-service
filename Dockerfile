FROM python:3.11-slim

# Install system dependencies for pdf2image (poppler) and Pillow
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

ENV NUM_WORKERS=2
ENV DB_PATH=/data/ocr_jobs.db
ENV OCR_BACKEND=huggingface

VOLUME ["/data"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

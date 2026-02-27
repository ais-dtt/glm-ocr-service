# GLM-OCR Worker Service

An async OCR processing service built with FastAPI that converts PDF and image files into markdown text using [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) — a 0.9B multimodal OCR model by ZhipuAI for text, table, and formula recognition.

## Architecture

```
                         ┌──────────────┐
                         │   Client     │
                         └──────┬───────┘
                                │
                         POST /ocr/submit
                         GET  /ocr/status/:id
                         GET  /ocr/result/:id
                                │
                         ┌──────▼───────┐
                         │   FastAPI    │
                         │   (routes)   │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │   SQLite DB  │
                         │  (aiosqlite) │
                         └──────┬───────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              ┌─────▼──┐ ┌─────▼──┐ ┌─────▼──┐
              │Worker 1│ │Worker 2│ │Worker N│
              └────┬───┘ └────┬───┘ └────┬───┘
                   │          │          │
              ┌────▼──────────▼──────────▼────┐
              │        OCR Backend            │
              │  (HuggingFace / Ollama)       │
              └───────────────────────────────┘
```

## Features

- Async PDF and image OCR processing
- Configurable worker pool for parallel processing
- SQLite database with async support (aiosqlite)
- PDF-to-image conversion with poppler
- Pluggable OCR backends (HuggingFace Gradio, Ollama)
- Job status tracking and result retrieval
- Pagination and filtering for job listings
- Docker support

## Prerequisites

- Python 3.11+
- `poppler-utils` (for PDF conversion)

Install poppler on Ubuntu/Debian:
```bash
sudo apt-get install poppler-utils
```

On macOS:
```bash
brew install poppler
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `NUM_WORKERS` | Number of OCR worker coroutines | `2` |
| `HF_TOKEN` | HuggingFace API token | `""` |
| `DB_PATH` | Path to SQLite database file | `./ocr_jobs.db` |
| `MAX_FILE_SIZE_MB` | Maximum upload file size in MB | `50` |
| `OCR_BACKEND` | OCR backend to use (`huggingface` or `ollama`) | `huggingface` |
| `OLLAMA_URL` | Ollama server URL | `""` |

You can set these via environment variables or a `.env` file.

## Running

```bash
uvicorn app.main:app --reload
```

The service starts at `http://localhost:8000`.

## Docker

Build:
```bash
docker build -t glm-ocr-service .
```

Run:
```bash
docker run -d \
  -p 8000:8000 \
  -v ocr-data:/data \
  -e HF_TOKEN=your_token_here \
  -e NUM_WORKERS=4 \
  glm-ocr-service
```

## API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

### Submit a File

Image:
```bash
curl -X POST http://localhost:8000/ocr/submit \
  -F "file=@document.png"
```

PDF:
```bash
curl -X POST http://localhost:8000/ocr/submit \
  -F "file=@document.pdf"
```

### Check Job Status

```bash
curl http://localhost:8000/ocr/status/{job_id}
```

### Get Job Result

```bash
curl http://localhost:8000/ocr/result/{job_id}
```

### List Jobs

```bash
curl "http://localhost:8000/ocr/jobs?status=queued&page=1&page_size=20"
```

### Delete a Job

```bash
curl -X DELETE http://localhost:8000/ocr/jobs/{job_id}
```

## Switching OCR Backends

### HuggingFace (default)

Uses the [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) model via HuggingFace Gradio API (prithivMLmods/GLM-OCR-Demo):
```bash
export OCR_BACKEND=huggingface
export HF_TOKEN=your_hf_token
```

### Ollama

Uses a local Ollama instance with the GLM-OCR model:
```bash
# Pull the model first
ollama pull glm-ocr

export OCR_BACKEND=ollama
export OLLAMA_URL=http://localhost:11434
```

## Running Tests

```bash
pytest tests/ -v
```

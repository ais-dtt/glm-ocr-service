from app.config import get_settings
from app.ocr_backends.base import OCRBackend
from app.ocr_backends.huggingface import HuggingFaceBackend
from app.ocr_backends.ollama import OllamaBackend


def get_ocr_backend() -> OCRBackend:
    settings = get_settings()
    if settings.OCR_BACKEND == "ollama":
        return OllamaBackend(ollama_url=settings.OLLAMA_URL)
    return HuggingFaceBackend(hf_token=settings.HF_TOKEN)

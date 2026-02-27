import base64
import logging

import httpx

from app.ocr_backends.base import OCRBackend, OCRProcessingError

logger = logging.getLogger(__name__)


class OllamaBackend(OCRBackend):
    """GLM-OCR backend via local Ollama server.

    Uses the glm-ocr model (zai-org/GLM-OCR, 0.9B params).
    Pull with: ollama pull glm-ocr
    """

    def __init__(self, ollama_url: str, model: str = "glm-ocr"):
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model

    async def process_image(self, image_bytes: bytes) -> str:
        if not self._ollama_url:
            raise OCRProcessingError(
                "Ollama backend not configured: set OLLAMA_URL env var"
            )

        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self._ollama_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
                        "images": [base64_str],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["response"]
        except httpx.HTTPError as e:
            raise OCRProcessingError(f"Ollama request failed: {e}") from e
        except KeyError:
            raise OCRProcessingError(
                "Unexpected Ollama response format: missing 'response' key"
            )

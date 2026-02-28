import base64
import logging

import httpx

from app.ocr_backends.base import OCRBackend, OCRProcessingError

logger = logging.getLogger(__name__)


class OllamaBackend(OCRBackend):
    """Self-hosted OCR backend via OpenAI-compatible API.

    Works with:
      - Ollama (ollama pull glm-ocr / deepseek-ocr)
      - vLLM (vllm serve deepseek-ai/DeepSeek-OCR)
      - Any OpenAI-compatible vision API

    Set OLLAMA_URL to the base URL (e.g. http://localhost:11434 for Ollama,
    http://localhost:8000 for vLLM).
    """

    def __init__(self, ollama_url: str, model: str = "deepseek-ai/DeepSeek-OCR"):
        self._base_url = ollama_url.rstrip("/")
        self._model = model

    async def process_image(self, image_bytes: bytes) -> str:
        if not self._base_url:
            raise OCRProcessingError(
                "Self-hosted backend not configured: set OLLAMA_URL env var"
            )

        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_str}"

        # Try OpenAI-compatible API first (/v1/chat/completions)
        # Works with both vLLM and Ollama
        try:
            return await self._call_openai_api(data_uri)
        except OCRProcessingError:
            raise
        except Exception as e:
            logger.info(f"OpenAI API failed ({e}), trying Ollama native API")

        # Fallback: Ollama native API (/api/generate)
        try:
            return await self._call_ollama_api(base64_str)
        except Exception as e:
            raise OCRProcessingError(f"All API attempts failed: {e}") from e

    async def _call_openai_api(self, data_uri: str) -> str:
        """Call OpenAI-compatible /v1/chat/completions (vLLM, Ollama)."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_uri},
                                    },
                                    {
                                        "type": "text",
                                        "text": "<|grounding|>Convert the document to markdown.",
                                    },
                                ],
                            }
                        ],
                        "max_tokens": 8192,
                        "temperature": 0.0,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            raise OCRProcessingError(
                f"OpenAI API request failed: {e}"
            ) from e
        except (KeyError, IndexError) as e:
            raise OCRProcessingError(
                f"Unexpected API response format: {e}"
            ) from e

    async def _call_ollama_api(self, base64_str: str) -> str:
        """Call Ollama native /api/generate."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
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
            raise OCRProcessingError(
                f"Ollama API request failed: {e}"
            ) from e
        except KeyError:
            raise OCRProcessingError(
                "Unexpected Ollama response: missing 'response' key"
            )

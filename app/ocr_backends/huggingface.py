import asyncio
import logging
import tempfile
import os

from app.ocr_backends.base import OCRBackend, OCRProcessingError

logger = logging.getLogger(__name__)


class HuggingFaceBackend(OCRBackend):
    def __init__(self, hf_token: str):
        self._hf_token = hf_token
        self._client = None

    def _get_client(self):
        if self._client is None:
            from gradio_client import Client
            self._client = Client(
                "prithivMLmods/GLM-OCR-Demo", hf_token=self._hf_token
            )
        return self._client

    async def process_image(self, image_bytes: bytes) -> str:
        tmp_path = None
        last_error = None

        for attempt in range(3):
            try:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                )
                tmp_path = tmp.name
                tmp.write(image_bytes)
                tmp.close()

                client = self._get_client()
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: client.predict(tmp_path, api_name="/predict"),
                )

                if isinstance(result, tuple):
                    result = result[0]
                if isinstance(result, dict):
                    result = result.get("text", str(result))

                return str(result)

            except Exception as e:
                last_error = e
                backoff = 2**attempt
                logger.warning(
                    f"OCR attempt {attempt + 1}/3 failed: {e}, "
                    f"retrying in {backoff}s"
                )
                if attempt < 2:
                    await asyncio.sleep(backoff)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        raise OCRProcessingError(
            f"OCR failed after 3 attempts: {last_error}"
        )

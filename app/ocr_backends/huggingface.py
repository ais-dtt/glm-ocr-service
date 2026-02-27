import asyncio
import logging
import tempfile
import os

from gradio_client import Client, handle_file

from app.ocr_backends.base import OCRBackend, OCRProcessingError

logger = logging.getLogger(__name__)


class HuggingFaceBackend(OCRBackend):
    """GLM-OCR backend via HuggingFace Space (prithivMLmods/GLM-OCR-Demo).

    Uses the zai-org/GLM-OCR model — a 0.9B multimodal OCR model
    for text, table, and formula recognition.
    """

    def __init__(self, hf_token: str):
        self._hf_token = hf_token
        self._client = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(
                "prithivMLmods/GLM-OCR-Demo", token=self._hf_token
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

                # GLM-OCR-Demo API: /process_image
                # Args: image (file), task ("Text"|"Formula"|"Table")
                # Returns: (raw_output: str, rendered_markdown: str)
                raw_output, rendered_md = await loop.run_in_executor(
                    None,
                    lambda: client.predict(
                        image=handle_file(tmp_path),
                        task="Text",
                        api_name="/process_image",
                    ),
                )

                return str(raw_output)

            except Exception as e:
                last_error = e
                backoff = 2**attempt
                logger.warning(
                    f"OCR attempt {attempt + 1}/3 failed: {e}, "
                    f"retrying in {backoff}s"
                )
                if attempt < 2:
                    self._client = None  # reset client on failure
                    await asyncio.sleep(backoff)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        raise OCRProcessingError(
            f"OCR failed after 3 attempts: {last_error}"
        )

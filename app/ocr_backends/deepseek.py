import asyncio
import logging
import tempfile
import os

from gradio_client import Client, handle_file

from app.ocr_backends.base import OCRBackend, OCRProcessingError

logger = logging.getLogger(__name__)


class DeepSeekBackend(OCRBackend):
    """DeepSeek-OCR-2 backend via HuggingFace Space.

    Uses deepseek-ai/DeepSeek-OCR-2 (3B params) via
    prithivMLmods/DeepSeek-OCR-2-Demo.

    Tasks: Markdown, Free OCR, OCR Image, Parse Figure
    Modes: Default, Quality, Fast, No Crop, Small
    """

    def __init__(self, hf_token: str, mode: str = "auto"):
        self._hf_token = hf_token
        self._mode = mode  # auto, text, table
        self._client = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(
                "prithivMLmods/DeepSeek-OCR-2-Demo", token=self._hf_token
            )
        return self._client

    async def _call_ocr(self, tmp_path: str, task: str = "Markdown") -> str:
        """Call DeepSeek-OCR-2 with a specific task."""
        client = self._get_client()
        loop = asyncio.get_event_loop()
        # Returns: (raw_text, markdown, token_info, image, gallery)
        result = await loop.run_in_executor(
            None,
            lambda: client.predict(
                image=handle_file(tmp_path),
                mode="Default",
                task=task,
                custom_prompt="",
                api_name="/process_image",
            ),
        )
        return str(result[0])

    async def process_image(self, image_bytes: bytes) -> str:
        """OCR with configurable mode.

        DeepSeek-OCR-2 'Markdown' task already outputs well-structured
        markdown with HTML tables (including rowspan/colspan) when needed.
        No two-pass required — single pass handles all content types.
        """
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

                result = await self._call_ocr(tmp_path, "Markdown")
                return result

            except Exception as e:
                last_error = e
                backoff = 2**attempt
                logger.warning(
                    f"DeepSeek OCR attempt {attempt + 1}/3 failed: {e}, "
                    f"retrying in {backoff}s"
                )
                if attempt < 2:
                    self._client = None
                    await asyncio.sleep(backoff)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        raise OCRProcessingError(
            f"DeepSeek OCR failed after 3 attempts: {last_error}"
        )

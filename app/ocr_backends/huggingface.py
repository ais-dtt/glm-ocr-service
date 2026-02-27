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

    async def _call_ocr(self, tmp_path: str, task: str) -> str:
        """Call GLM-OCR with a specific task mode."""
        client = self._get_client()
        loop = asyncio.get_event_loop()
        raw_output, rendered_md = await loop.run_in_executor(
            None,
            lambda: client.predict(
                image=handle_file(tmp_path),
                task=task,
                api_name="/process_image",
            ),
        )
        return str(raw_output)

    async def process_image(self, image_bytes: bytes) -> str:
        """Two-pass OCR: Text mode for markdown + Table mode for HTML tables.

        Pass 1 (Text): Clean markdown for all content.
        Pass 2 (Table): HTML with rowspan/colspan preserved.

        Returns both combined — markdown text with an HTML table section appended.
        Downstream consumers can use whichever format suits them.
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

                # Pass 1: Text mode — always run
                text_result = await self._call_ocr(tmp_path, "Text")

                # Pass 2: Table mode — run if text has table-like content
                has_table = ("|" in text_result and ("---" in text_result or "| :" in text_result))

                if has_table:
                    try:
                        table_html = await self._call_ocr(tmp_path, "Table")
                    except Exception as e:
                        logger.warning(f"Table pass failed, text only: {e}")
                        table_html = None

                    if table_html and "<table" in table_html:
                        # Append HTML tables section after the markdown
                        return (
                            text_result
                            + "\n\n<!-- HTML tables with rowspan/colspan -->\n"
                            + table_html
                        )

                return text_result

            except Exception as e:
                last_error = e
                backoff = 2**attempt
                logger.warning(
                    f"OCR attempt {attempt + 1}/3 failed: {e}, "
                    f"retrying in {backoff}s"
                )
                if attempt < 2:
                    self._client = None
                    await asyncio.sleep(backoff)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        raise OCRProcessingError(
            f"OCR failed after 3 attempts: {last_error}"
        )

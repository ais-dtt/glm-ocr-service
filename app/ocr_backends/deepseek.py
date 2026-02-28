import asyncio
import base64
import io
import logging
import re
import tempfile
import os

from gradio_client import Client, handle_file
from PIL import Image

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

    @staticmethod
    def _fix_latex_dollars(text: str) -> str:
        r"""Fix DeepSeek's LaTeX escaping of dollar signs.

        DeepSeek interprets $45.2M as inline LaTeX, producing:
          \(45.2M  in one cell and  \)12.8M  in the next.

        Fix: replace \( or \) followed by digits/currency content with $.
        Preserve real LaTeX like \(x^2 + y^2\) which has paired delimiters
        with math operators inside.
        """
        import re

        # \( followed by digit, comma, or dot → currency $ sign
        text = re.sub(r"\\\((?=[\d,.])", "$", text)

        # \) followed by digit, comma, or dot → currency $ sign
        text = re.sub(r"\\\)(?=[\d,.])", "$", text)

        # \) at end of content that looks like currency
        # e.g. "15,700/mo (Staging)\)" → "$15,700/mo (Staging)"
        # The \) here closes a fake LaTeX that started with \( elsewhere
        text = re.sub(r"\\\)(?=\s*<)", "$", text)

        return text

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
                result = self._fix_latex_dollars(result)
                result = self._extract_inline_images(result, image_bytes)
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

    @staticmethod
    def _extract_inline_images(text: str, page_image_bytes: bytes) -> str:
        """Extract image regions detected by DeepSeek and embed as base64.

        DeepSeek outputs placeholder tags like:
          <img src="imgs/img_in_image_box_X1_Y1_X2_Y2.jpg">

        This method:
        1. Parses bounding box coordinates from the filename
        2. Crops that region from the original page image
        3. Replaces the placeholder with a base64 data URI
        """
        img_pattern = re.compile(
            r'<img\s+src="imgs/img[^"]*?_(\d+)_(\d+)_(\d+)_(\d+)\.\w+"[^>]*>'
        )

        matches = list(img_pattern.finditer(text))
        if not matches:
            return text

        try:
            page_img = Image.open(io.BytesIO(page_image_bytes))
        except Exception as e:
            logger.warning(f"Failed to open page image for cropping: {e}")
            return text

        for match in reversed(matches):  # reverse to preserve positions
            x1, y1, x2, y2 = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            )

            try:
                # Clamp to image bounds
                x1 = max(0, min(x1, page_img.width))
                y1 = max(0, min(y1, page_img.height))
                x2 = max(0, min(x2, page_img.width))
                y2 = max(0, min(y2, page_img.height))

                if x2 <= x1 or y2 <= y1:
                    continue

                cropped = page_img.crop((x1, y1, x2, y2))
                buf = io.BytesIO()
                cropped.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                replacement = (
                    f'<img src="data:image/png;base64,{b64}" '
                    f'alt="extracted image ({x1},{y1})-({x2},{y2})" />'
                )
                text = text[:match.start()] + replacement + text[match.end():]
            except Exception as e:
                logger.warning(
                    f"Failed to crop image region ({x1},{y1})-({x2},{y2}): {e}"
                )

        return text

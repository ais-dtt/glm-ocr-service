from abc import ABC, abstractmethod


class OCRBackend(ABC):
    """Abstract base class for OCR backends.

    All backends must implement process_image which takes raw PNG/image bytes
    and returns the extracted text as a markdown string.
    """

    @abstractmethod
    async def process_image(self, image_bytes: bytes) -> str:
        """Process an image and return OCR result as markdown text.

        Args:
            image_bytes: Raw image bytes (PNG format preferred)

        Returns:
            Markdown-formatted OCR result string

        Raises:
            OCRProcessingError: If OCR fails after retries
        """
        ...


class OCRProcessingError(Exception):
    """Raised when OCR processing fails."""
    pass

import asyncio
import logging
import uuid
from typing import List

from app.ocr_client import get_ocr_backend

logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(self):
        self._workers: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self._worker_ids: List[str] = []
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def start(self, num_workers: int) -> None:
        """Start N worker coroutines."""
        self._stop_event.clear()
        for i in range(num_workers):
            worker_id = f"worker-{i + 1}-{uuid.uuid4().hex[:8]}"
            self._worker_ids.append(worker_id)
            task = asyncio.create_task(self._run_worker(worker_id))
            self._workers.append(task)
        logger.info(f"Started {num_workers} OCR workers")

    async def stop(self) -> None:
        """Signal workers to stop and wait for them."""
        self._stop_event.set()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._worker_ids.clear()
        logger.info("All OCR workers stopped")

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    @property
    def active_workers(self) -> int:
        return self._active_count

    async def _run_worker(self, worker_id: str) -> None:
        """Main worker loop: poll DB, claim jobs, process, save results."""
        from app.database import AsyncSessionLocal
        from app.crud import (
            claim_page_job,
            get_next_queued_page,
            update_page_job_result,
            check_and_update_parent_status,
        )
        from app.ocr_backends.base import OCRProcessingError

        ocr_backend = get_ocr_backend()
        logger.info(f"[{worker_id}] Worker started")

        while not self._stop_event.is_set():
            try:
                async with AsyncSessionLocal() as db:
                    page_job = await get_next_queued_page(db)
                    if page_job is None:
                        await asyncio.sleep(1.0)
                        continue

                    claimed = await claim_page_job(db, page_job.id, worker_id)
                    if not claimed:
                        continue  # Another worker got it

                    logger.info(
                        f"[{worker_id}] Processing page job {page_job.id} "
                        f"(page {page_job.page_number})"
                    )

                    async with self._lock:
                        self._active_count += 1

                    try:
                        markdown_text = await ocr_backend.process_image(
                            page_job.image_data
                        )
                        await update_page_job_result(
                            db, page_job.id, markdown_text, "completed"
                        )
                        await check_and_update_parent_status(
                            db, page_job.parent_job_id
                        )
                        logger.info(
                            f"[{worker_id}] Completed page job {page_job.id}"
                        )
                    except OCRProcessingError as e:
                        logger.error(
                            f"[{worker_id}] OCR failed for page job "
                            f"{page_job.id}: {e}"
                        )
                        await update_page_job_result(
                            db, page_job.id, None, "failed", str(e)
                        )
                        await check_and_update_parent_status(
                            db, page_job.parent_job_id
                        )
                    except Exception as e:
                        logger.error(
                            f"[{worker_id}] Unexpected error: {e}",
                            exc_info=True,
                        )
                        await update_page_job_result(
                            db, page_job.id, None, "failed", str(e)
                        )
                        await check_and_update_parent_status(
                            db, page_job.parent_job_id
                        )
                    finally:
                        async with self._lock:
                            self._active_count -= 1

            except Exception as e:
                logger.error(
                    f"[{worker_id}] Worker loop error: {e}", exc_info=True
                )
                await asyncio.sleep(2.0)

        logger.info(f"[{worker_id}] Worker stopped")


# Singleton instance
worker_manager = WorkerManager()

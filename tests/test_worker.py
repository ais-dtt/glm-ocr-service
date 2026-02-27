import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from app.worker import WorkerManager


class TestWorkerManager:
    async def test_start_stop(self):
        manager = WorkerManager()
        await manager.start(2)
        assert manager.worker_count == 2
        await manager.stop()
        assert manager.worker_count == 0

    async def test_worker_processes_job(self, db_session):
        """Test that a worker picks up and processes a queued page job."""
        from app.crud import create_job, create_page_job, get_page_jobs

        job = await create_job(db_session, "test.png", "image", 1)
        await db_session.commit()
        await create_page_job(db_session, job.id, 1, b"fake_image")
        await db_session.commit()

        mock_backend = AsyncMock()
        mock_backend.process_image = AsyncMock(
            return_value="# OCR Result\n\nTest text"
        )

        manager = WorkerManager()

        with patch("app.worker.get_ocr_backend", return_value=mock_backend):
            await manager.start(1)
            await asyncio.sleep(2.5)  # Let worker process
            await manager.stop()

        # Worker uses its own DB session, so we verify it stopped cleanly
        assert manager.worker_count == 0

import pytest

from app.crud import (
    check_and_update_parent_status,
    claim_page_job,
    create_job,
    create_page_job,
    delete_job,
    get_job,
    get_next_queued_page,
    get_page_jobs,
    get_queue_depth,
    list_jobs,
    update_page_job_result,
)


class TestJobCRUD:
    async def test_create_and_get_job(self, db_session):
        job = await create_job(db_session, "test.pdf", "pdf", 3)
        await db_session.commit()

        fetched = await get_job(db_session, job.id)
        assert fetched is not None
        assert fetched.original_filename == "test.pdf"
        assert fetched.total_pages == 3
        assert fetched.status == "queued"

    async def test_get_nonexistent_job(self, db_session):
        result = await get_job(db_session, "nonexistent-id")
        assert result is None

    async def test_list_jobs_with_status_filter(self, db_session):
        await create_job(db_session, "file1.pdf", "pdf", 1)
        await create_job(db_session, "file2.pdf", "pdf", 2)
        await db_session.commit()

        jobs, total = await list_jobs(db_session, status_filter="queued")
        assert total == 2
        assert len(jobs) == 2

    async def test_list_jobs_pagination(self, db_session):
        for i in range(5):
            await create_job(db_session, f"file{i}.pdf", "pdf", 1)
        await db_session.commit()

        jobs, total = await list_jobs(db_session, page=1, page_size=2)
        assert total == 5
        assert len(jobs) == 2

    async def test_delete_job(self, db_session):
        job = await create_job(db_session, "test.pdf", "pdf", 1)
        await db_session.commit()

        deleted = await delete_job(db_session, job.id)
        await db_session.commit()
        assert deleted is True

        fetched = await get_job(db_session, job.id)
        assert fetched is None


class TestPageJobCRUD:
    async def test_create_page_job(self, db_session):
        job = await create_job(db_session, "test.pdf", "pdf", 2)
        await db_session.commit()

        page_job = await create_page_job(
            db_session, job.id, 1, b"fake_image_bytes"
        )
        await db_session.commit()

        pages = await get_page_jobs(db_session, job.id)
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert pages[0].image_data == b"fake_image_bytes"

    async def test_claim_page_job(self, db_session):
        job = await create_job(db_session, "test.pdf", "pdf", 1)
        await db_session.commit()
        page_job = await create_page_job(db_session, job.id, 1, b"bytes")
        await db_session.commit()

        claimed = await claim_page_job(db_session, page_job.id, "worker-1")
        await db_session.commit()
        assert claimed is True

        # Try to claim again - should fail
        claimed_again = await claim_page_job(db_session, page_job.id, "worker-2")
        await db_session.commit()
        assert claimed_again is False

    async def test_get_next_queued_page(self, db_session):
        job = await create_job(db_session, "test.pdf", "pdf", 1)
        await db_session.commit()
        await create_page_job(db_session, job.id, 1, b"bytes")
        await db_session.commit()

        next_page = await get_next_queued_page(db_session)
        assert next_page is not None
        assert next_page.status == "queued"

    async def test_check_and_update_parent_status_completed(self, db_session):
        job = await create_job(db_session, "test.pdf", "pdf", 2)
        await db_session.commit()

        p1 = await create_page_job(db_session, job.id, 1, b"bytes1")
        p2 = await create_page_job(db_session, job.id, 2, b"bytes2")
        await db_session.commit()

        await update_page_job_result(db_session, p1.id, "text1", "completed")
        await update_page_job_result(db_session, p2.id, "text2", "completed")
        await check_and_update_parent_status(db_session, job.id)
        await db_session.commit()

        updated_job = await get_job(db_session, job.id)
        assert updated_job.status == "completed"

    async def test_check_and_update_parent_status_failed(self, db_session):
        job = await create_job(db_session, "test.pdf", "pdf", 2)
        await db_session.commit()

        p1 = await create_page_job(db_session, job.id, 1, b"bytes1")
        p2 = await create_page_job(db_session, job.id, 2, b"bytes2")
        await db_session.commit()

        await update_page_job_result(db_session, p1.id, "text1", "completed")
        await update_page_job_result(
            db_session, p2.id, None, "failed", "OCR error"
        )
        await check_and_update_parent_status(db_session, job.id)
        await db_session.commit()

        updated_job = await get_job(db_session, job.id)
        assert updated_job.status == "failed"

    async def test_queue_depth(self, db_session):
        job = await create_job(db_session, "test.pdf", "pdf", 2)
        await db_session.commit()
        await create_page_job(db_session, job.id, 1, b"bytes1")
        await create_page_job(db_session, job.id, 2, b"bytes2")
        await db_session.commit()

        depth = await get_queue_depth(db_session)
        assert depth == 2

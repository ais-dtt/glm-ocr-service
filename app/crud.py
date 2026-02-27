import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import OcrJob, OcrPageJob

logger = logging.getLogger(__name__)


async def create_job(
    db: AsyncSession,
    original_filename: str,
    file_type: str,
    total_pages: int,
) -> OcrJob:
    job = OcrJob(
        id=str(uuid.uuid4()),
        original_filename=original_filename,
        file_type=file_type,
        total_pages=total_pages,
        status="queued",
    )
    db.add(job)
    await db.flush()
    return job


async def create_page_job(
    db: AsyncSession,
    parent_job_id: Optional[str] = None,
    page_number: int = 1,
    image_data: bytes = b"",
    *,
    job_id: Optional[str] = None,
) -> OcrPageJob:
    # Accept both parent_job_id and job_id for compatibility
    actual_parent_id = parent_job_id or job_id
    if not actual_parent_id:
        raise ValueError("parent_job_id or job_id is required")
    page_job = OcrPageJob(
        id=str(uuid.uuid4()),
        parent_job_id=actual_parent_id,
        page_number=page_number,
        image_data=image_data,
        status="queued",
    )
    db.add(page_job)
    await db.flush()
    return page_job


async def get_job(db: AsyncSession, job_id: str) -> Optional[OcrJob]:
    result = await db.execute(select(OcrJob).where(OcrJob.id == job_id))
    return result.scalar_one_or_none()


async def get_page_jobs(db: AsyncSession, parent_job_id: str) -> List[OcrPageJob]:
    result = await db.execute(
        select(OcrPageJob)
        .where(OcrPageJob.parent_job_id == parent_job_id)
        .order_by(OcrPageJob.page_number)
    )
    return list(result.scalars().all())


async def get_next_queued_page(db: AsyncSession) -> Optional[OcrPageJob]:
    """Get the next queued page job (does not claim it yet)."""
    result = await db.execute(
        select(OcrPageJob)
        .where(OcrPageJob.status == "queued")
        .order_by(OcrPageJob.created_at)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def claim_page_job(
    db: AsyncSession, page_job_id: str, worker_id: str
) -> bool:
    """Atomically claim a page job. Returns True if successfully claimed."""
    result = await db.execute(
        update(OcrPageJob)
        .where(OcrPageJob.id == page_job_id, OcrPageJob.status == "queued")
        .values(
            status="processing",
            worker_id=worker_id,
            updated_at=datetime.now(timezone.utc),
        )
        .returning(OcrPageJob.id)
    )
    await db.flush()
    return result.scalar_one_or_none() is not None


async def update_page_job_result(
    db: AsyncSession,
    page_job_id: str,
    markdown_text: Optional[str],
    status: str,
    error_message: Optional[str] = None,
) -> None:
    await db.execute(
        update(OcrPageJob)
        .where(OcrPageJob.id == page_job_id)
        .values(
            markdown_text=markdown_text,
            status=status,
            error_message=error_message,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await db.flush()


async def update_job_status(db: AsyncSession, job_id: str, status: str) -> None:
    await db.execute(
        update(OcrJob)
        .where(OcrJob.id == job_id)
        .values(status=status, updated_at=datetime.now(timezone.utc))
    )
    await db.flush()


async def check_and_update_parent_status(
    db: AsyncSession, parent_job_id: str
) -> None:
    """Check all page jobs for a parent and update parent status if all done."""
    result = await db.execute(
        select(OcrPageJob.status).where(
            OcrPageJob.parent_job_id == parent_job_id
        )
    )
    statuses = [row[0] for row in result.fetchall()]

    if not statuses:
        return

    if all(s == "completed" for s in statuses):
        await update_job_status(db, parent_job_id, "completed")
    elif any(s == "failed" for s in statuses) and all(
        s in ("completed", "failed") for s in statuses
    ):
        await update_job_status(db, parent_job_id, "failed")
    elif any(s == "processing" for s in statuses):
        await update_job_status(db, parent_job_id, "processing")


async def list_jobs(
    db: AsyncSession,
    status_filter: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    *,
    status: Optional[str] = None,
) -> Tuple[List[OcrJob], int]:
    # Accept both status_filter and status for compatibility
    effective_status = status_filter or status
    query = select(OcrJob)
    count_query = select(func.count(OcrJob.id))

    if effective_status:
        query = query.where(OcrJob.status == effective_status)
        count_query = count_query.where(OcrJob.status == effective_status)

    query = (
        query.order_by(OcrJob.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    result = await db.execute(query)
    count_result = await db.execute(count_query)

    jobs = list(result.scalars().all())
    total = count_result.scalar_one()
    return jobs, total


async def delete_job(db: AsyncSession, job_id: str) -> bool:
    result = await db.execute(
        delete(OcrJob).where(OcrJob.id == job_id).returning(OcrJob.id)
    )
    await db.flush()
    return result.scalar_one_or_none() is not None


async def get_queue_depth(db: AsyncSession) -> int:
    result = await db.execute(
        select(func.count(OcrPageJob.id)).where(OcrPageJob.status == "queued")
    )
    return result.scalar_one()

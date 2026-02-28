import logging
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.crud import (
    create_job,
    create_page_job,
    delete_job,
    get_job,
    get_page_jobs,
    get_queue_depth,
    list_jobs,
)
from app.database import get_db
import re

from app.schemas import (
    HealthResponse,
    JobListResponse,
    JobResultResponse,
    JobStatusResponse,
    JobSubmitResponse,
    PageResult,
    Section,
)

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}

ocr_router = APIRouter(prefix="/ocr")
health_router = APIRouter()


@ocr_router.post("/submit", response_model=JobSubmitResponse)
async def submit_job(file: UploadFile, db: AsyncSession = Depends(get_db)):
    settings = get_settings()

    # Validate file extension
    filename = file.filename or ""
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Validate content type
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'.",
        )

    # Read file and validate size
    file_bytes = await file.read()
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum of {settings.MAX_FILE_SIZE_MB} MB.",
        )

    # Convert to page images
    pages: list[bytes] = []
    file_type = "image"

    if ext == ".pdf":
        file_type = "pdf"
        try:
            from pdf2image import convert_from_bytes

            pil_images = convert_from_bytes(file_bytes, dpi=150)
            for img in pil_images:
                buf = BytesIO()
                img.save(buf, "PNG")
                pages.append(buf.getvalue())
        except Exception as e:
            logger.error("Failed to convert PDF: %s", e)
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")
    else:
        pages.append(file_bytes)

    total_pages = len(pages)

    # Create job and page jobs
    job = await create_job(
        db,
        original_filename=filename,
        total_pages=total_pages,
        file_type=file_type,
    )

    for i, page_bytes in enumerate(pages):
        await create_page_job(
            db,
            job_id=job.job_id,
            page_number=i + 1,
            image_data=page_bytes,
        )

    logger.info("Created job %s with %d pages", job.job_id, total_pages)

    return JobSubmitResponse(
        job_id=job.job_id,
        total_pages=total_pages,
        message=f"Job submitted successfully with {total_pages} page(s).",
    )


@ocr_router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    page_jobs = await get_page_jobs(db, job_id)
    completed_pages = sum(1 for p in page_jobs if p.status == "completed")
    failed_pages = sum(1 for p in page_jobs if p.status == "failed")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total_pages=job.total_pages,
        completed_pages=completed_pages,
        failed_pages=failed_pages,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _parse_sections(pages: list[PageResult]) -> list[Section]:
    """Parse markdown headings into structured sections across all pages."""
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[Section] = []

    for page in pages:
        if not page.markdown_text:
            continue

        text = page.markdown_text
        matches = list(heading_re.finditer(text))

        if not matches:
            # No headings — entire page is one section
            content = text.strip()
            if content:
                sections.append(Section(
                    heading="(untitled)",
                    level=0,
                    page=page.page_number,
                    content=content,
                ))
            continue

        # Content before first heading
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(Section(
                heading="(untitled)",
                level=0,
                page=page.page_number,
                content=pre,
            ))

        # Each heading and its content until the next heading
        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            sections.append(Section(
                heading=heading,
                level=level,
                page=page.page_number,
                content=content,
            ))

    return sections


@ocr_router.get("/result/{job_id}", response_model=JobResultResponse)
async def get_job_result(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    page_jobs = await get_page_jobs(db, job_id)
    pages = sorted(
        [
            PageResult(
                page_number=p.page_number,
                markdown_text=p.markdown_text,
                status=p.status,
            )
            for p in page_jobs
        ],
        key=lambda p: p.page_number,
    )

    sections = _parse_sections(pages)

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        pages=pages,
        sections=sections,
        total_pages=job.total_pages,
        total_sections=len(sections),
    )


@ocr_router.get("/jobs", response_model=JobListResponse)
async def list_all_jobs(
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    if page_size > 100:
        raise HTTPException(status_code=400, detail="page_size must be <= 100")
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")

    jobs, total = await list_jobs(db, status=status, page=page, page_size=page_size)

    return JobListResponse(
        jobs=[
            {
                "job_id": j.job_id,
                "original_filename": j.original_filename,
                "status": j.status,
                "total_pages": j.total_pages,
                "file_type": j.file_type,
                "created_at": j.created_at,
            }
            for j in jobs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@ocr_router.delete("/jobs/{job_id}", status_code=204)
async def delete_job_endpoint(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    await delete_job(db, job_id)
    return Response(status_code=204)


@health_router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    settings = get_settings()
    worker_count = settings.NUM_WORKERS
    active_workers = 0

    try:
        from app.worker import worker_manager

        worker_count = worker_manager.worker_count
        active_workers = worker_manager.active_workers
    except (ImportError, AttributeError):
        pass

    queue_depth_val = await get_queue_depth(db)

    return HealthResponse(
        status="ok",
        worker_count=worker_count,
        active_workers=active_workers,
        queue_depth=queue_depth_val,
        db_path=settings.DB_PATH,
    )

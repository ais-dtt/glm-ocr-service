from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class JobSubmitResponse(BaseModel):
    job_id: str
    total_pages: int
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    total_pages: int
    completed_pages: int
    failed_pages: int
    created_at: datetime
    updated_at: datetime


class PageResult(BaseModel):
    page_number: int
    markdown_text: Optional[str]
    status: str


class Section(BaseModel):
    heading: str
    level: int  # 1=h1, 2=h2, 3=h3, etc.
    page: int
    content: str  # full content under this heading (including sub-sections)


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    pages: List[PageResult]
    total_pages: int


class JobSectionsResponse(BaseModel):
    job_id: str
    status: str
    document_name: str
    sections: List[Section]
    total_sections: int


class JobListItem(BaseModel):
    job_id: str
    original_filename: str
    status: str
    total_pages: int
    file_type: str
    created_at: datetime


class JobListResponse(BaseModel):
    jobs: List[JobListItem]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str
    worker_count: int
    active_workers: int
    queue_depth: int
    db_path: str

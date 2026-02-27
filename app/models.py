import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import String, Integer, LargeBinary, ForeignKey, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class OcrJob(Base):
    __tablename__ = "ocr_jobs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)  # "pdf" or "image"
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String, default="queued"
    )  # queued/processing/completed/failed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    pages: Mapped[List["OcrPageJob"]] = relationship(
        "OcrPageJob", back_populates="parent_job", cascade="all, delete-orphan"
    )

    @property
    def job_id(self) -> str:
        return self.id


class OcrPageJob(Base):
    __tablename__ = "ocr_page_jobs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    parent_job_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("ocr_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    image_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    markdown_text: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="queued")
    worker_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    parent_job: Mapped["OcrJob"] = relationship("OcrJob", back_populates="pages")

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import init_db
from app.routes import health_router, ocr_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger.info("Starting GLM-OCR Worker Service")

    await init_db()

    settings = get_settings()
    try:
        from app.worker import worker_manager

        await worker_manager.start(settings.NUM_WORKERS)
        logger.info("Workers started: %d", settings.NUM_WORKERS)
    except ImportError:
        logger.warning("Worker module not available, skipping worker startup")

    yield

    logger.info("Shutting down GLM-OCR Worker Service")
    try:
        from app.worker import worker_manager

        await worker_manager.stop()
    except ImportError:
        pass


app = FastAPI(
    title="GLM-OCR Worker Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ocr_router)
app.include_router(health_router)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

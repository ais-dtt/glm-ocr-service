from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    NUM_WORKERS: int = 2
    HF_TOKEN: str = ""
    DB_PATH: str = "./ocr_jobs.db"
    MAX_FILE_SIZE_MB: int = 50
    OLLAMA_URL: str = ""
    OCR_BACKEND: str = "huggingface"
    OCR_MODE: str = "auto"  # "auto" (text+table two-pass), "text", "table"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://vendorwatch:vendorwatch@localhost:5432/vendorwatch"
    SYNC_DATABASE_URL: str = "postgresql://vendorwatch:vendorwatch@localhost:5432/vendorwatch"
    MODELS_DIR: str = "/models"
    FEEDBACK_RETRAIN_THRESHOLD: int = 50
    CUSUM_K: float = 0.5
    CUSUM_H: float = 5.0
    IF_CONTAMINATION: float = 0.05
    IF_ESTIMATORS: int = 200
    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

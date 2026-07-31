from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "FastAPI App"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_COLLECTION: str = "templates"
    HYBRID_ALPHA: float = 0.7
    POSTGRES_DSN: str = "postgresql://appuser:apppassword@localhost:5432/appdb"
    STORAGE_ADAPTER: str = "postgres"
    OUTBOX_POLL_INTERVAL: float = 5.0
    OUTBOX_MAX_RETRIES: int = 5


settings = Settings()

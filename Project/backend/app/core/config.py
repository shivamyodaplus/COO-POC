from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "FastAPI App"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"


settings = Settings()

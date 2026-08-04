from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "FastAPI App"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_COLLECTION: str = "templates"
    PACD_MILVUS_COLLECTION: str = "pacd_coo_documents"
    HYBRID_ALPHA: float = 0.7
    POSTGRES_DSN: str = "postgresql://appuser:apppassword@localhost:5432/appdb"
    STORAGE_ADAPTER: str = "postgres"
    OUTBOX_POLL_INTERVAL: float = 5.0
    OUTBOX_MAX_RETRIES: int = 5
    # vLLM endpoints — injected by docker-compose via host.docker.internal.
    # Defaults point to localhost for direct (non-Docker) development.
    QWEN_TEXT_BASE_URL: str = "http://localhost:8001/v1"
    QWEN_VL_BASE_URL: str = "http://localhost:8002/v1"
    # VLLM inference endpoints
    VLLM_TEXT_MODEL: str = "qwen3-4b-awq"
    VLLM_VISION_MODEL: str = "qwen3-vl-4b"
    VLLM_API_KEY: str = "vllm-local"
    VLLM_TEMPERATURE: float = 0.0
    # None = omit max_tokens from the request so the model uses its full context window.
    VLLM_MAX_TOKENS: int | None = None

    # BGE-M3 text embedding (PACD / COO cross-reference)
    BGE_M3_MODEL: str = "BAAI/bge-m3"
    BGE_M3_USE_FP16: bool = True
    # Force a specific device for BGE-M3.  Leave empty for auto-detection via
    # torch.cuda.is_available().  Set to 'cuda' on ARM64/Jetson where PyTorch
    # is a CPU build yet the NVIDIA Container Toolkit exposes the GPU.
    BGE_M3_DEVICE: str = ""

    # PACD chunking strategy
    CHUNK_WINDOW: int = 5   # KV pairs per grouped chunk
    CHUNK_STRIDE: int = 3   # stride between grouped chunks


settings = Settings()

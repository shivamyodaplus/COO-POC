from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env files (if present)

# Resolves to /app inside Docker (WORKDIR) and Project/backend on the host.
_BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load backend-local .env first, then project-root .env (which takes
        # precedence). Both are resolved from the package location so the path
        # is correct regardless of the working directory.  In Docker, env vars
        # are injected by docker-compose so missing files are silently ignored.
        env_file=[
            str(_BASE_DIR / ".env"),          # Project/backend/.env
            str(_BASE_DIR.parent / ".env"),   # Project/.env  (overrides above)
        ],
        env_file_encoding="utf-8",
        extra="ignore",
    )

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
    # LLM provider selection: "local" (VLLM) or "bedrock" (AWS Bedrock)
    LLM_PROVIDER: str = "local"

    # VLLM inference endpoints
    VLLM_TEXT_MODEL: str = "qwen3-4b-awq"
    VLLM_VISION_MODEL: str = "qwen3-vl-4b"
    VLLM_API_KEY: str = "vllm-local"
    VLLM_TEMPERATURE: float = 0.0
    # None = omit max_tokens from the request so the model uses its full context window.
    VLLM_MAX_TOKENS: int | None = None

    # AWS Bedrock settings (used when LLM_PROVIDER="bedrock")
    # Authentication uses a Bedrock short-term API key via the OpenAI-compatible
    # Chat Completions endpoint (https://bedrock-mantle.<region>.api.aws/v1).
    # Generate a key at: https://console.aws.amazon.com/bedrock/home#/api-keys/short-term/create
    BEDROCK_API_KEY: str = ""
    BEDROCK_TEXT_MODEL: str = "qwen.qwen3-32b-v1:0"
    # BEDROCK_VISION_MODEL: str = "nvidia.nemotron-nano-12b-v2"
    # BEDROCK_VISION_MODEL: str = "anthropic.claude-sonnet-5"
    BEDROCK_VISION_MODEL: str = "anthropic.claude-sonnet-5"
    BEDROCK_REGION: str = "ap-south-1"
    # Bedrock Converse requires an explicit maxTokens; None is rejected or
    # defaults to the model maximum, which combined with a large prompt can
    # exceed the total context window.  Override via BEDROCK_MAX_TOKENS in .env.
    # BEDROCK_MAX_TOKENS: int = 120000
    BEDROCK_MAX_TOKENS: int = 30000

    # BGE-M3 text embedding (PACD / COO cross-reference)
    BGE_M3_MODEL: str = "BAAI/bge-m3"
    BGE_M3_USE_FP16: bool = True
    # Force a specific device for BGE-M3.  Leave empty for auto-detection via
    # torch.cuda.is_available().  Set to 'cuda' on ARM64/Jetson where PyTorch
    # is a CPU build yet the NVIDIA Container Toolkit exposes the GPU.
    BGE_M3_DEVICE: str = ""

    # Local storage root — auto-detected from the package location:
    # /app inside Docker (WORKDIR), Project/backend on the host.
    # Override via LOCAL_STORAGE_DIR in .env if needed.
    LOCAL_STORAGE_DIR: str = str(_BASE_DIR)

    # PACD chunking strategy
    CHUNK_WINDOW: int = 5   # KV pairs per grouped chunk
    CHUNK_STRIDE: int = 3   # stride between grouped chunks

    # COO cross-reference: Milvus hits fetched per field query
    CROSS_REF_TOP_K: int = 3

    # ── Map-Reduce RAG pipeline (new COO verification flow) ──────────────────
    # Name of the new granular PACD chunk collection.
    PACD_CHUNK_COLLECTION: str = "pacd_document_chunks"
    # Max line items per table chunk — splits large tables into multiple chunks
    # so the embedding and LLM context stay manageable (20 items ≈ 4 KB text).
    MAX_TABLE_ITEMS_PER_CHUNK: int = 20
    # Milvus ANN candidates fetched per individual query (before RRF fusion).
    RAG_RETRIEVAL_TOP_K: int = 10
    # Final unique chunks returned after RRF fusion and deduplication.
    RAG_FINAL_TOP_K: int = 8
    # Max sibling table chunks fetched per logical document during table isolation.
    TABLE_ISOLATION_MAX_CHUNKS: int = 10

    # COO template retrieval & confirmation
    # How many candidate templates to fetch from Milvus.
    # The confirmation node tries them one-by-one in similarity order, so
    # raising this gives a better chance of finding the right template at the
    # cost of additional Vision LLM calls in the worst case.
    COO_TEMPLATE_TOP_K: int = 3
    # Max templates sent to the Vision LLM per call.
    # Total images per call = 1 (COO) + COO_TEMPLATE_CONFIRM_BATCH.
    # Set to 1 for LLMs that accept only 2 images; raise for models with a
    # higher image-count limit to reduce the number of round-trips.
    COO_TEMPLATE_CONFIRM_BATCH: int = 1


settings = Settings()

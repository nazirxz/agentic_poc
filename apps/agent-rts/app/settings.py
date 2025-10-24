from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    DOMAIN: str = "RTS"
    CATEGORY_FILTER: str = "rokan_technical_standard"  # Default: no filtering, use all categories
    REFUSAL_TEXT: str = "Tidak ditemukan dalam RTS."
    TOP_K: int = 16
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 8
    THRESHOLD: float = 0.55
    MAX_CONTEXT: int = 8
    STYLE: str = "teknis-formal"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:0.6b"
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3"

    MILVUS_CONNECTION_URI: str = "http://localhost:19530"
    MILVUS_COLLECTION_NAME: str = "rokan_technical_standard"
    RERANK_URL: str = "http://localhost:8082"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

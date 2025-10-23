from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    DOMAIN: str = "RTS"
    CATEGORY_FILTER: str = "RTS"
    REFUSAL_TEXT: str = "Tidak ditemukan dalam RTS."
    TOP_K: int = 16
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 8
    THRESHOLD: float = 0.55
    MAX_CONTEXT: int = 8
    STYLE: str = "teknis-formal"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MILVUS_RAG_URL: str = "http://localhost:19530"
    RERANK_URL: str = "http://localhost:8082"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

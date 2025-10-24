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
    
    # Configurable search parameters
    MAX_PASSAGES: int = 4
    MAX_TEXT_LENGTH: int = 500
    MAX_CONTEXT_LENGTH: int = 2000
    MIN_RELEVANCE_SCORE: float = -0.5
    MIN_TEXT_LENGTH: int = 30
    
    # Configurable penalties and boosts
    NON_TECHNICAL_PENALTY: float = 0.3
    SHORT_TEXT_PENALTY: float = 0.2
    HTML_PENALTY: float = 0.8
    KEYWORD_BOOST: float = 0.5
    TECHNICAL_TERM_BOOST: float = 0.2
    RTS_TERM_BOOST: float = 0.1
    
    # Configurable timeouts
    LLM_TIMEOUT: int = 30
    MAX_RETRIES: int = 2

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

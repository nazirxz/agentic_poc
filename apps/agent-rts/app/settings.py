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
    
    # BGE-M3 Local Model Paths
    BGE_M3_MODEL_PATH: str = "/local/urbuddy_rag/Milvus_Prod/llm_models/bge-m3"
    BGE_RERANKER_MODEL_PATH: str = "/local/urbuddy_rag/Milvus_Prod/llm_models/bge-reranker-v2-m3"
    
    # Use BGE-M3 for both dense and sparse embeddings
    USE_BGE_M3_HYBRID: bool = True  # Enable hybrid search with dense + sparse
    
    MILVUS_CONNECTION_URI: str = "http://localhost:19530"
    MILVUS_COLLECTION_NAME: str = "rokan_technical_standard"
    
    # Enable reranker for better results
    USE_RERANKER: bool = True
    RERANKER_TOP_K: int = 20  # Get more candidates before reranking
    FINAL_TOP_K: int = 5  # Final results after reranking
    
    # Configurable search parameters
    MAX_PASSAGES: int = 4
    MAX_TEXT_LENGTH: int = 500
    MAX_CONTEXT_LENGTH: int = 2000
    MIN_RELEVANCE_SCORE: float = 0.5  # Increased from -0.5 to filter irrelevant results
    MIN_TEXT_LENGTH: int = 30
    
    # Configurable reranking weights (can be tuned per domain)
    VECTOR_SCORE_WEIGHT: float = 0.3
    TEXT_SCORE_WEIGHT: float = 2.0
    METADATA_SCORE_WEIGHT: float = 1.0
    
    # BM25 parameters (can be tuned)
    BM25_K1_KEYWORD: float = 1.5  # Term frequency saturation for keywords
    BM25_K1_CONTENT: float = 1.2  # Term frequency saturation for content tokens
    
    # Keyword extraction parameters
    MAX_KEYWORDS: int = 8
    KEYWORD_FREQ_WEIGHT: float = 2.0
    KEYWORD_POSITION_WEIGHT: float = 0.5
    
    # Configurable timeouts
    LLM_TIMEOUT: int = 30
    MAX_RETRIES: int = 2

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

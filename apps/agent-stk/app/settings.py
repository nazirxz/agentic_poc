from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    DOMAIN: str = "STK"
    CATEGORY_FILTER: str = "STK"
    REFUSAL_TEXT: str = "Tidak ditemukan dalam STK."
    TOP_K: int = 16
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 8
    THRESHOLD: float = 0.55
    MAX_CONTEXT: int = 8
    STYLE: str = "ringkas-berbutir"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:0.6b"
    
    # BGE-M3 Local Model Paths
    BGE_M3_MODEL_PATH: str = "/local/urbuddy_rag/Milvus_Prod/llm_models/bge-m3"
    BGE_RERANKER_MODEL_PATH: str = "/local/urbuddy_rag/Milvus_Prod/llm_models/bge-reranker-v2-m3"
    
    # Use BGE-M3 for both dense and sparse embeddings
    USE_BGE_M3_HYBRID: bool = True
    
    MILVUS_CONNECTION_URI: str = "http://localhost:19530"
    MILVUS_COLLECTIONS: dict = {
        "tko": "tko",           # Tata Kerja Organisasi - prosedur
        "tki": "tki",           # Tata Kerja Individual - instruksi kerja
        "tkpa": "tkpa",         # Tata Kerja Penggunaan Alat - instruksi kerja
        "pedoman": "pedoman"    # Manual
    }
    RERANK_URL: str = "http://localhost:8082"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

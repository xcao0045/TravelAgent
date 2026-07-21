import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    bailian_api_key: str = ""
    amap_api_key: str = ""
    llm_model: str = "qwen-max"
    embedding_model: str = "text-embedding-v3"
    chroma_persist_dir: str = "./storage"
    history_dir: str = "./data/history"
    top_k_preferences: int = 5
    top_k_cases: int = 3
    similarity_threshold: float = 0.45
    # Child chunk (向量化, 存入 ChromaDB)
    child_chunk_size: int = 500
    child_chunk_overlap: int = 50
    # Parent chunk (仅存储文本, 检索命中 Child 后回查返回)
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    # Ensemble retrieval
    search_type: str = "ensemble"     # "ensemble" | "vector" | "bm25"
    rrf_k: int = 60                   # RRF 公式中的 k 常量
    # 向后兼容别名 (已废弃, 请使用 child_chunk_size / child_chunk_overlap)
    chunk_size: int = 500
    chunk_overlap: int = 50

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            bailian_api_key=os.getenv("BAILIAN_API_KEY", ""),
            amap_api_key=os.getenv("AMAP_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "qwen-max"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./storage"),
            history_dir=os.getenv("HISTORY_DIR", "./data/history"),
            top_k_preferences=int(os.getenv("TOP_K_PREFERENCES", "5")),
            top_k_cases=int(os.getenv("TOP_K_CASES", "3")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.45")),
            child_chunk_size=int(os.getenv("CHILD_CHUNK_SIZE", "500")),
            child_chunk_overlap=int(os.getenv("CHILD_CHUNK_OVERLAP", "50")),
            parent_chunk_size=int(os.getenv("PARENT_CHUNK_SIZE", "2000")),
            parent_chunk_overlap=int(os.getenv("PARENT_CHUNK_OVERLAP", "200")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            search_type=os.getenv("SEARCH_TYPE", "ensemble"),
            rrf_k=int(os.getenv("RRF_K", "60")),
        )

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
    similarity_threshold: float = 0.7
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
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        )

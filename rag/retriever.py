from langchain_core.documents import Document
from rag.vector_store import VectorStoreManager


class DualRetriever:
    """统一的雙庫檢索接口"""

    def __init__(self, vector_store: VectorStoreManager, similarity_threshold: float = 0.7):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def retrieve_preferences(
        self, query: str, category: str | None = None, k: int = 5
    ) -> list[Document]:
        """檢索偏好庫，可選按 category 過濾。只返回相似度 >= threshold 的結果。"""
        collection = self.vector_store.get_preferences_collection()
        if category:
            results = collection.similarity_search_with_relevance_scores(
                query, k=k, filter={"category": category}
            )
        else:
            results = collection.similarity_search_with_relevance_scores(query, k=k)
        return [doc for doc, score in results if score >= self.similarity_threshold]

    def retrieve_cases(self, query: str, k: int = 3) -> list[Document]:
        """檢索案例庫。只返回相似度 >= threshold 的結果。"""
        collection = self.vector_store.get_cases_collection()
        results = collection.similarity_search_with_relevance_scores(query, k=k)
        return [doc for doc, score in results if score >= self.similarity_threshold]

    def retrieve_both(
        self,
        query: str,
        preferences_category: str | None = None,
        k_prefs: int = 5,
        k_cases: int = 3,
    ) -> dict:
        """同時檢索兩個庫，返回 {"preferences": [...], "cases": [...]}"""
        prefs = self.retrieve_preferences(query, category=preferences_category, k=k_prefs)
        cases = self.retrieve_cases(query, k=k_cases)
        return {"preferences": prefs, "cases": cases}

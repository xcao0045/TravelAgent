from langchain_core.documents import Document
from rag.vector_store import VectorStoreManager


class DualRetriever:
    """统一的双库检索接口 — Parent-Child 映射版。

    检索命中子 Chunk 后，通过 parent_id 回查 ParentDocStore，
    返回父 Chunk 以提供更完整的上下文。同一 Parent 的去重自动处理。
    """

    def __init__(self, vector_store: VectorStoreManager, similarity_threshold: float = 0.45):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    # ── 内部方法 ────────────────────────────────────────────

    def _resolve_to_parents(self, scored_docs: list[tuple[Document, float]]) -> list[Document]:
        """子 Chunk → 父 Chunk 映射 + 去重。

        工作流:
        1. 按相似度阈值过滤
        2. 对每个 child，查 parent_id → parent_store
        3. 找到 parent → 返回 parent；找不到 → 回退返回 child
        4. 同一 parent_id 只保留第一次出现（保持检索得分排序）
        """
        seen: set[str] = set()
        resolved: list[Document] = []

        for child, score in scored_docs:
            if score < self.similarity_threshold:
                continue
            parent_id = child.metadata.get("parent_id", "")
            if parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent = self.vector_store.parent_store.get(parent_id)
                if parent:
                    resolved.append(parent)
                else:
                    resolved.append(child)
            elif not parent_id and id(child) not in seen:
                # 兼容旧数据：无 parent_id 的 chunk，按 Python id 去重
                seen.add(str(id(child)))
                resolved.append(child)

        return resolved

    # ── 公开接口 ─────────────────────────────────────────────

    def retrieve_preferences(
        self, query: str, category: str | None = None, k: int = 5
    ) -> list[Document]:
        collection = self.vector_store.get_preferences_collection()
        if category:
            results = collection.similarity_search_with_relevance_scores(
                query, k=k, filter={"category": category}
            )
        else:
            results = collection.similarity_search_with_relevance_scores(query, k=k)
        return self._resolve_to_parents(results)

    def retrieve_cases(self, query: str, k: int = 3) -> list[Document]:
        collection = self.vector_store.get_cases_collection()
        results = collection.similarity_search_with_relevance_scores(query, k=k)
        return self._resolve_to_parents(results)

    def retrieve_both(
        self,
        query: str,
        preferences_category: str | None = None,
        k_prefs: int = 5,
        k_cases: int = 3,
    ) -> dict:
        prefs = self.retrieve_preferences(query, category=preferences_category, k=k_prefs)
        cases = self.retrieve_cases(query, k=k_cases)
        return {"preferences": prefs, "cases": cases}

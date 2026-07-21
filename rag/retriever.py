from langchain_core.documents import Document
from rag.vector_store import VectorStoreManager


def rrf_fuse(
    vec_ranked: list[Document],
    bm25_ranked: list[tuple[str, float]],
    k: int = 60,
) -> list[Document]:
    """RRF (Reciprocal Rank Fusion) — 融合向量和 BM25 两路 Child 排名。

    vec_ranked: 向量检索返回的 Document 列表（已按相似度降序排列）
    bm25_ranked: BM25 检索返回的 [(chroma_id_or_parent_id, score), ...]（已按得分降序）
    k: RRF 常量，默认 60。k 越小排名权重越大。

    返回按 RRF 得分降序排列的 Document 列表（Child 层，需后续 _resolve_to_parents）。
    """
    if not vec_ranked and not bm25_ranked:
        return []
    if not bm25_ranked:
        return list(vec_ranked)
    if not vec_ranked:
        return []

    # 为 Document 建立稳定的 ID：优先 parent_id，其次 Python id
    def _doc_key(doc: Document) -> str:
        return doc.metadata.get("parent_id", "") or str(id(doc))

    # 向量路排名: {doc_key: rank}
    vec_ranks: dict[str, int] = {}
    for rank, doc in enumerate(vec_ranked, 1):
        vec_ranks[_doc_key(doc)] = rank

    # BM25 路排名: {doc_key: rank}
    bm25_ranks: dict[str, int] = {}
    # 构建 bm25_id → rank 映射
    for rank, (bm25_id, _) in enumerate(bm25_ranked, 1):
        bm25_ranks[bm25_id] = rank

    # 收集所有出现过的 Document
    seen: dict[str, Document] = {}
    for doc in vec_ranked:
        key = _doc_key(doc)
        if key not in seen:
            seen[key] = doc
    # BM25 结果中可能有向量路没有的 doc — 这种情况无法回查,
    # 因为 Document 对象不在 vec_ranked 中。后续 Ensemble Retriever 负责解决。

    # 计算 RRF 得分
    scores: list[tuple[Document, float]] = []
    for key, doc in seen.items():
        rrf_score = 0.0
        if key in vec_ranks:
            rrf_score += 1.0 / (k + vec_ranks[key])
        if key in bm25_ranks:
            rrf_score += 1.0 / (k + bm25_ranks[key])
        scores.append((doc, rrf_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores]


class DualRetriever:
    """统一的双库检索接口 — Ensemble (向量+BM25) + Parent-Child 映射版。

    支持三种检索模式:
    - "ensemble": 向量 + BM25 双路 → RRF 融合 → Parent 映射 (默认)
    - "vector":   纯向量检索 → Parent 映射
    - "bm25":     纯 BM25 检索 → Parent 映射
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        similarity_threshold: float = 0.45,
        search_type: str = "ensemble",
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.search_type = search_type
        self.rrf_k = rrf_k

    # ── 内部: Parent 映射 ────────────────────────────────────

    def _resolve_to_parents(self, scored_docs: list[tuple[Document, float]]) -> list[Document]:
        """子 Chunk → 父 Chunk 映射 + 去重。"""
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
                seen.add(str(id(child)))
                resolved.append(child)

        return resolved

    def _resolve_children_to_parents(self, children: list[Document]) -> list[Document]:
        """将已排好序的 Children 列表映射为 Parents（去重）。"""
        seen: set[str] = set()
        resolved: list[Document] = []
        for child in children:
            parent_id = child.metadata.get("parent_id", "")
            if parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent = self.vector_store.parent_store.get(parent_id)
                resolved.append(parent if parent else child)
            elif not parent_id:
                resolved.append(child)
        return resolved

    # ── 内部: 向量检索 ───────────────────────────────────────

    def _vector_search(self, collection_name: str, query: str,
                       category: str | None, k: int) -> list[tuple[Document, float]]:
        if collection_name == "preferences":
            coll = self.vector_store.get_preferences_collection()
        else:
            coll = self.vector_store.get_cases_collection()
        if category:
            return coll.similarity_search_with_relevance_scores(
                query, k=k, filter={"category": category})
        return coll.similarity_search_with_relevance_scores(query, k=k)

    # ── 内部: BM25 检索 + 回查 Document ──────────────────────

    def _bm25_search(self, collection_name: str, query: str,
                     k: int) -> list[tuple[Document, float]]:
        """BM25 检索 → 从 ChromaDB 回查 Document → 返回 [(Document, bm25_score), ...]。"""
        bm25_results = self.vector_store.bm25_index.search(collection_name, query, k=k)
        if not bm25_results:
            return []
        # 从 ChromaDB 批量回查 Document
        if collection_name == "preferences":
            coll = self.vector_store.get_preferences_collection()
        else:
            coll = self.vector_store.get_cases_collection()
        ids_to_fetch = [cid for cid, _ in bm25_results]
        chroma_data = coll.get(ids=ids_to_fetch)
        # 构建 id → Document 映射
        id_to_doc: dict[str, Document] = {}
        for idx, doc_id in enumerate(chroma_data["ids"]):
            id_to_doc[doc_id] = Document(
                page_content=chroma_data["documents"][idx] if chroma_data["documents"] else "",
                metadata=chroma_data["metadatas"][idx] if chroma_data["metadatas"] else {},
            )
        # 保持 BM25 排序
        return [(id_to_doc[cid], score) for cid, score in bm25_results if cid in id_to_doc]

    # ── 内部: Ensemble 调度 ──────────────────────────────────

    def _ensemble_search(self, collection_name: str, query: str,
                         category: str | None, k: int) -> list[Document]:
        """向量 + BM25 双路 → RRF 融合 → Parent 映射。"""
        # 两路各取 k*2 候选，给 RRF 更大的融合空间
        fetch_k = k * 2
        vec_results = self._vector_search(collection_name, query, category, k=fetch_k)
        bm25_results = self._bm25_search(collection_name, query, k=fetch_k)

        # 过滤向量路低于阈值的结果（BM25 不适用阈值）
        vec_filtered = [doc for doc, score in vec_results if score >= self.similarity_threshold]

        # RRF 融合
        fused_children = rrf_fuse(vec_filtered, bm25_results, k=self.rrf_k)

        # Parent 映射 + 截断
        return self._resolve_children_to_parents(fused_children)[:k]

    def _vector_only(self, collection_name: str, query: str,
                     category: str | None, k: int) -> list[Document]:
        results = self._vector_search(collection_name, query, category, k=k)
        return self._resolve_to_parents(results)[:k]

    def _bm25_only(self, collection_name: str, query: str,
                   category: str | None, k: int) -> list[Document]:
        results = self._bm25_search(collection_name, query, k=k)
        # 对 category 做后置过滤
        if category:
            results = [(doc, s) for doc, s in results
                       if doc.metadata.get("category") == category]
        return self._resolve_to_parents(results)[:k]

    # ── 公开接口 ─────────────────────────────────────────────

    def retrieve_preferences(
        self, query: str, category: str | None = None, k: int = 5
    ) -> list[Document]:
        if self.search_type == "vector":
            return self._vector_only("preferences", query, category, k)
        elif self.search_type == "bm25":
            return self._bm25_only("preferences", query, category, k)
        else:  # ensemble (default)
            return self._ensemble_search("preferences", query, category, k)

    def retrieve_cases(self, query: str, k: int = 3) -> list[Document]:
        if self.search_type == "vector":
            return self._vector_only("cases", query, None, k)
        elif self.search_type == "bm25":
            return self._bm25_only("cases", query, None, k)
        else:  # ensemble (default)
            return self._ensemble_search("cases", query, None, k)

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

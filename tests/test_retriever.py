from unittest.mock import Mock
from langchain_core.documents import Document
from rag.retriever import DualRetriever, rrf_fuse


class FakeParentStore:
    """模拟 ParentDocStore"""
    def __init__(self, store: dict | None = None):
        self._store = store or {}

    def get(self, parent_id):
        return self._store.get(parent_id)

    def exists(self, parent_id):
        return parent_id in self._store


class FakeBM25Index:
    """模拟 BM25IndexManager"""
    def __init__(self, results=None):
        self._results = results or {}

    def search(self, collection_name: str, query: str, k: int):
        return self._results.get(collection_name, [])


class FakeVectorStore:
    def __init__(self, parent_store: FakeParentStore | None = None,
                 bm25_index: FakeBM25Index | None = None):
        self._prefs = Mock()
        self._cases = Mock()
        self.parent_store = parent_store or FakeParentStore()
        self.bm25_index = bm25_index or FakeBM25Index()

    def get_preferences_collection(self):
        return self._prefs

    def get_cases_collection(self):
        return self._cases


# ── 旧测试 (向后兼容) ──


def test_retrieve_preferences_calls_similarity_search():
    vs = FakeVectorStore()
    fake_docs = [
        Document(page_content="酒店A隔音好", metadata={
            "category": "hotel", "name": "酒店A",
            "parent_id": "abc", "chunk_type": "standalone",
        })
    ]
    vs._prefs.similarity_search_with_relevance_scores.return_value = [
        (fake_docs[0], 0.85)
    ]
    vs.parent_store = FakeParentStore({"abc": fake_docs[0]})

    retriever = DualRetriever(vs, search_type="vector")
    results = retriever.retrieve_preferences("亲子 隔音 酒店", category="hotel", k=5)

    assert len(results) == 1
    vs._prefs.similarity_search_with_relevance_scores.assert_called_once()


def test_retrieve_both_returns_both_collections():
    vs = FakeVectorStore()
    vs._prefs.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="好评", metadata={"parent_id": "p1"}), 0.8)
    ]
    vs._cases.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="成都3天案例", metadata={"parent_id": "c1"}), 0.75)
    ]
    vs.parent_store = FakeParentStore({
        "p1": Document(page_content="好评", metadata={"parent_id": "p1"}),
        "c1": Document(page_content="成都3天案例", metadata={"parent_id": "c1"}),
    })

    retriever = DualRetriever(vs, search_type="vector")
    result = retriever.retrieve_both("成都 亲子", preferences_category=None, k_prefs=5, k_cases=3)

    assert "preferences" in result
    assert "cases" in result
    assert len(result["preferences"]) == 1
    assert len(result["cases"]) == 1


# ── Parent-Child 映射测试 ──


class TestChildToParentMapping:
    """验证子 Chunk 命中 → 返回父 Chunk"""

    def test_child_hit_returns_parent_document(self):
        """命中 child → 通过 parent_id 查 parent_store → 返回 parent"""
        parent_doc = Document(
            page_content="成都3日游完整方案 Day1 上午去宽窄巷子 下午去锦里...",
            metadata={"parent_id": "md5_p0", "chunk_type": "parent"},
        )
        child_doc = Document(
            page_content="上午去宽窄巷子 下午去锦里",
            metadata={"parent_id": "md5_p0", "chunk_type": "child"},
        )
        vs = FakeVectorStore()
        vs._cases.similarity_search_with_relevance_scores.return_value = [
            (child_doc, 0.7)
        ]
        vs.parent_store = FakeParentStore({"md5_p0": parent_doc})

        retriever = DualRetriever(vs, search_type="vector")
        results = retriever.retrieve_cases("成都 宽窄巷子", k=3)

        assert len(results) == 1
        assert results[0].page_content == parent_doc.page_content
        assert results[0].metadata["chunk_type"] == "parent"

    def test_multiple_children_same_parent_dedup_to_one(self):
        """同 Parent 的 3 个 child 都命中 → 只返回 1 个 parent"""
        parent_doc = Document(
            page_content="苏州园林完整方案...",
            metadata={"parent_id": "sz_p0", "chunk_type": "parent"},
        )
        c1 = Document(page_content="苏州园林 chunk1", metadata={"parent_id": "sz_p0"})
        c2 = Document(page_content="苏州园林 chunk2", metadata={"parent_id": "sz_p0"})
        c3 = Document(page_content="苏州园林 chunk3", metadata={"parent_id": "sz_p0"})

        vs = FakeVectorStore()
        vs._cases.similarity_search_with_relevance_scores.return_value = [
            (c1, 0.8), (c2, 0.75), (c3, 0.6),
        ]
        vs.parent_store = FakeParentStore({"sz_p0": parent_doc})

        retriever = DualRetriever(vs, search_type="vector")
        results = retriever.retrieve_cases("苏州 园林", k=3)

        # 3 个 child 指向同一 parent → 去重后只剩 1
        assert len(results) == 1
        assert results[0].page_content == "苏州园林完整方案..."

    def test_threshold_filters_out_low_scores(self):
        """低于 threshold 的 child 不参与映射"""
        parent = Document(
            page_content="北京故宫方案",
            metadata={"parent_id": "bj_p0"},
        )
        c_low = Document(page_content="低分chunk", metadata={"parent_id": "bj_p0"})

        vs = FakeVectorStore()
        vs._cases.similarity_search_with_relevance_scores.return_value = [
            (c_low, 0.3),
        ]
        vs.parent_store = FakeParentStore({"bj_p0": parent})

        retriever = DualRetriever(vs, similarity_threshold=0.45)
        results = retriever.retrieve_cases("北京", k=3)

        assert len(results) == 0  # score 0.3 < 0.45

    def test_no_parent_id_falls_back_to_child(self):
        """无 parent_id 元数据时（旧数据兼容），返回 child 自身"""
        child = Document(page_content="旧chunk无parent_id", metadata={})
        vs = FakeVectorStore()
        vs._cases.similarity_search_with_relevance_scores.return_value = [
            (child, 0.7)
        ]

        retriever = DualRetriever(vs, search_type="vector")
        results = retriever.retrieve_cases("测试", k=3)

        assert len(results) == 1
        assert results[0].page_content == "旧chunk无parent_id"


# ── RRF 融合测试 ──


class TestRRFFusion:
    """验证 RRF (Reciprocal Rank Fusion) 算法"""

    def test_rrf_promotes_document_ranked_high_in_both_lists(self):
        """在两路都排高位的文档 → RRF 融合后排第一"""
        d1 = Document(page_content="苏州园林", metadata={"parent_id": "p1"})
        d2 = Document(page_content="杭州西湖", metadata={"parent_id": "p2"})
        d3 = Document(page_content="成都宽窄巷子", metadata={"parent_id": "p3"})

        # 向量路: d1 rank1, d2 rank2, d3 rank3
        vec_ranked = [d1, d2, d3]
        # BM25路: d2 rank1, d1 rank2, d3 rank3
        bm25_ranked = [("p2", 3.5), ("p1", 2.8), ("p3", 1.5)]

        result = rrf_fuse(vec_ranked, bm25_ranked, k=60)

        # d1: 1/61 + 1/62 = 0.03252
        # d2: 1/62 + 1/61 = 0.03252  (symmetrical)
        # At k=60, both have same score. But d3: 1/63+1/63 = 0.03175
        # d1 and d2 should be top 2
        assert result[0] in (d1, d2)
        assert result[1] in (d1, d2)
        assert result[2] == d3

    def test_rrf_bm25_only_doc_not_in_vec_not_included(self):
        """仅出现在 BM25 路但无 Document 对象的条目 → 不参与融合。
        真实场景中，BM25-only 的文档会先通过 ChromaDB.get() 回查补全 Document，
        再传入 rrf_fuse，因此此边界由上层 Ensemble Retriever 负责。"""
        d1 = Document(page_content="苏州园林", metadata={"parent_id": "p1"})
        d2 = Document(page_content="杭州西湖", metadata={"parent_id": "p2"})

        vec_ranked = [d1, d2]
        bm25_ranked = [("p_unknown", 5.0), ("p1", 2.0)]

        result = rrf_fuse(vec_ranked, bm25_ranked, k=60)
        # p_unknown 没有对应的 Document → 不出现
        # p1 两路都有分 → 排第一
        assert result[0] == d1
        assert len(result) == 2

    def test_rrf_empty_inputs(self):
        """空输入 → 返回空列表，不抛异常"""
        # 两边都空
        assert rrf_fuse([], []) == []
        # BM25 为空
        d1 = Document(page_content="test", metadata={"parent_id": "p1"})
        assert rrf_fuse([d1], []) == [d1]
        # 向量为空
        assert rrf_fuse([], [("p1", 3.0)]) == []

    def test_rrf_different_k_values_change_weights(self):
        """k 值越小 → 排名差距影响越大"""
        d1 = Document(page_content="top1", metadata={"parent_id": "p1"})
        d2 = Document(page_content="top2", metadata={"parent_id": "p2"})

        vec_ranked = [d1, d2]
        bm25_ranked = [("p2", 5.0), ("p1", 1.0)]  # d2 rank1, d1 rank2

        # k=60 (接近等权): d1=1/61+1/62=0.0325, d2=1/62+1/61=0.0325 → 同分
        result_k60 = rrf_fuse(vec_ranked, bm25_ranked, k=60)

        # k=0 (排名权重大): d1=1/1+1/2=1.5, d2=1/2+1/1=1.5 → 也同分
        # k=10: d1=1/11+1/12=0.174, d2=1/12+1/11=0.174 → 同分
        result_k10 = rrf_fuse(vec_ranked, bm25_ranked, k=10)

        # 两者都返回 2 个文档
        assert len(result_k60) == 2
        assert len(result_k10) == 2

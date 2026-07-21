from unittest.mock import Mock
from langchain_core.documents import Document
from rag.retriever import DualRetriever


class FakeParentStore:
    """模拟 ParentDocStore"""
    def __init__(self, store: dict | None = None):
        self._store = store or {}

    def get(self, parent_id):
        return self._store.get(parent_id)

    def exists(self, parent_id):
        return parent_id in self._store


class FakeVectorStore:
    def __init__(self, parent_store: FakeParentStore | None = None):
        self._prefs = Mock()
        self._cases = Mock()
        self.parent_store = parent_store or FakeParentStore()

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

    retriever = DualRetriever(vs)
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

    retriever = DualRetriever(vs)
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

        retriever = DualRetriever(vs)
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

        retriever = DualRetriever(vs)
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

        retriever = DualRetriever(vs)
        results = retriever.retrieve_cases("测试", k=3)

        assert len(results) == 1
        assert results[0].page_content == "旧chunk无parent_id"

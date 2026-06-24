from unittest.mock import Mock
from langchain_core.documents import Document
from rag.retriever import DualRetriever


class FakeVectorStore:
    def __init__(self):
        self._prefs = Mock()
        self._cases = Mock()

    def get_preferences_collection(self):
        return self._prefs

    def get_cases_collection(self):
        return self._cases


def test_retrieve_preferences_calls_similarity_search():
    vs = FakeVectorStore()
    fake_docs = [
        Document(page_content="酒店A隔音好", metadata={"category": "hotel", "name": "酒店A"})
    ]
    vs._prefs.similarity_search_with_relevance_scores.return_value = [
        (fake_docs[0], 0.85)
    ]

    retriever = DualRetriever(vs)
    results = retriever.retrieve_preferences("亲子 隔音 酒店", category="hotel", k=5)

    assert len(results) == 1
    vs._prefs.similarity_search_with_relevance_scores.assert_called_once()


def test_retrieve_both_returns_both_collections():
    vs = FakeVectorStore()
    vs._prefs.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="好评", metadata={}), 0.8)
    ]
    vs._cases.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="成都3天案例", metadata={}), 0.75)
    ]

    retriever = DualRetriever(vs)
    result = retriever.retrieve_both("成都 亲子", preferences_category=None, k_prefs=5, k_cases=3)

    assert "preferences" in result
    assert "cases" in result
    assert len(result["preferences"]) == 1
    assert len(result["cases"]) == 1

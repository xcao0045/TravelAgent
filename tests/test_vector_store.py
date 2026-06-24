import tempfile
import shutil
from unittest.mock import patch, Mock
from rag.vector_store import VectorStoreManager


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 1024


def test_create_collections():
    tmpdir = tempfile.mkdtemp()
    try:
        embeddings = FakeEmbeddings()
        manager = VectorStoreManager(persist_dir=tmpdir, embeddings=embeddings)

        prefs = manager.get_preferences_collection()
        cases = manager.get_cases_collection()

        assert prefs._collection.name == "user_preferences"
        assert cases._collection.name == "travel_cases"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_and_query_preferences():
    from langchain_core.documents import Document

    tmpdir = tempfile.mkdtemp()
    try:
        embeddings = FakeEmbeddings()
        manager = VectorStoreManager(persist_dir=tmpdir, embeddings=embeddings)

        doc = Document(
            page_content="酒店A 隔音效果好，适合亲子出行，卫生条件优秀",
            metadata={"category": "hotel", "name": "酒店A", "tags": ["亲子", "隔音好"], "rating": 4.5},
        )
        ids = manager.add_to_preferences([doc])
        assert len(ids) == 1

        results = manager.get_preferences_collection().similarity_search(
            query="亲子 隔音 酒店",
            k=3,
        )
        assert len(results) >= 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

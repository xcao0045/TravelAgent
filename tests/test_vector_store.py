import tempfile
import shutil
from unittest.mock import patch, Mock
from langchain_core.documents import Document
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


# ── Parent-Child chunking 测试 ──


class TestParentChildChunking:
    """验证长文档切分为 Parent-Child，短文档走 standalone"""

    def test_short_doc_becomes_standalone(self):
        """≤ child_chunk_size 的短文 → standalone chunk"""
        tmpdir = tempfile.mkdtemp()
        try:
            manager = VectorStoreManager(
                persist_dir=tmpdir,
                embeddings=FakeEmbeddings(),
                child_chunk_size=500,
                parent_chunk_size=2000,
            )
            doc = Document(
                page_content="这家酒店很棒",
                metadata={"category": "hotel", "name": "好评酒店"},
            )
            ids = manager.add_to_preferences([doc])
            assert len(ids) == 1

            # 验证 ChromaDB 中的 metadata
            coll = manager.get_preferences_collection()
            stored = coll.get()
            meta = stored["metadatas"][0]
            assert meta["chunk_type"] == "standalone"
            assert "parent_id" in meta
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_long_doc_splits_to_parents_and_children(self):
        """> child_chunk_size 的长文 → 产生多个 parent 和更多 child"""
        tmpdir = tempfile.mkdtemp()
        try:
            manager = VectorStoreManager(
                persist_dir=tmpdir,
                embeddings=FakeEmbeddings(),
                child_chunk_size=200,   # 极小, 强制多切
                parent_chunk_size=600,  # 小, 但可容纳 2-3 个 child
            )
            # 构造一篇 ~2000 字的中文文档
            content = (
                "杭州三日游。第一天：西湖游览。" + "断桥残雪是西湖十景之一。" * 6 +
                "第二天：灵隐寺。" + "灵隐寺是千年古刹。" * 6 +
                "第三天：龙井村。" + "龙井茶闻名天下。" * 6
            )
            doc = Document(
                page_content=content,
                metadata={"destination": "杭州", "days": 3},
            )
            ids = manager.add_to_cases([doc])
            # 至少产生 2 个 child
            assert len(ids) >= 2

            # 验证 children 有 parent_id
            coll = manager.get_cases_collection()
            stored = coll.get()
            assert all(m["chunk_type"] == "child" for m in stored["metadatas"])
            assert all("parent_id" in m for m in stored["metadatas"])
            assert all("source_md5" in m for m in stored["metadatas"])

            # 验证 parent_store 有对应的 parent
            for meta in stored["metadatas"]:
                pid = meta["parent_id"]
                parent = manager.parent_store.get(pid)
                assert parent is not None
                assert parent.metadata["chunk_type"] == "parent"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_all_children_from_same_source_share_source_md5(self):
        """同一原始文档的所有 children 共享 source_md5"""
        tmpdir = tempfile.mkdtemp()
        try:
            manager = VectorStoreManager(
                persist_dir=tmpdir,
                embeddings=FakeEmbeddings(),
                child_chunk_size=200,
                parent_chunk_size=600,
            )
            content = "成都游记。" + "宽窄巷子很好玩。" * 20
            doc = Document(page_content=content, metadata={"destination": "成都"})
            manager.add_to_cases([doc])

            coll = manager.get_cases_collection()
            stored = coll.get()
            source_md5s = {m["source_md5"] for m in stored["metadatas"]}
            assert len(source_md5s) == 1  # 同一个 source
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_standalone_docs_stored_in_parent_store(self):
        """Short documents get stored in parent_store with chunk_type='standalone'"""
        tmpdir = tempfile.mkdtemp()
        try:
            manager = VectorStoreManager(
                persist_dir=tmpdir,
                embeddings=FakeEmbeddings(),
                child_chunk_size=500,
            )
            doc = Document(
                page_content="苏州园林真的很美",
                metadata={"category": "attraction", "name": "拙政园"},
            )
            ids = manager.add_to_preferences([doc])
            assert len(ids) == 1

            # 验证 parent_store 中存了 standalone
            pid = ids[0]
            stored_meta = manager.get_preferences_collection().get()["metadatas"][0]
            parent_id = stored_meta["parent_id"]
            parent = manager.parent_store.get(parent_id)
            assert parent is not None
            assert parent.page_content == "苏州园林真的很美"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_delete_by_source_also_clears_parent_store(self):
        """删除 source_md5 同时清除 ChromaDB 和 parent_store"""
        tmpdir = tempfile.mkdtemp()
        try:
            manager = VectorStoreManager(
                persist_dir=tmpdir,
                embeddings=FakeEmbeddings(),
                child_chunk_size=500,
            )
            doc = Document(
                page_content="短评",
                metadata={"category": "hotel", "name": "测试酒店"},
            )
            manager.add_to_preferences([doc])
            stored = manager.get_preferences_collection().get()
            sm5 = stored["metadatas"][0]["source_md5"]
            parent_id = stored["metadatas"][0]["parent_id"]

            # 删除前，parent_store 有数据
            assert manager.parent_store.get(parent_id) is not None

            manager.delete_by_source(sm5, "preferences")

            # 删除后，ChromaDB 无数据
            remaining = manager.get_preferences_collection().get()
            assert len(remaining["ids"]) == 0

            # 删除后，parent_store 也清空
            assert manager.parent_store.get(parent_id) is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestSemanticBoundarySplitting:
    """验证中文语义边界切分"""

    def test_child_not_cut_mid_sentence(self):
        """小于 child_chunk_size 的句子不被硬性截断"""
        tmpdir = tempfile.mkdtemp()
        try:
            manager = VectorStoreManager(
                persist_dir=tmpdir,
                embeddings=FakeEmbeddings(),
                child_chunk_size=500,
                parent_chunk_size=2000,
            )
            content = "第一天：游览西湖。\n\n西湖位于杭州市区，是中国十大风景名胜之一。\n\n断桥残雪是西湖最著名的景点，每到冬季雪后，桥面若隐若现。"
            doc = Document(page_content=content, metadata={"destination": "杭州"})
            ids = manager.add_to_cases([doc])
            # 文本 ~150 字，远小于 child_chunk_size=500，应不切为 standalone
            assert len(ids) == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

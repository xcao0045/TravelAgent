"""TDD Round 1: ParentDocStore 测试"""
import tempfile
import os
from langchain_core.documents import Document
from rag.parent_store import ParentDocStore


class TestParentStoreBasic:
    """最基本的写入/读取/存在性检查"""

    def test_put_and_get_returns_same_document(self):
        """存入一个 Parent Document → 读取回来, page_content 和 metadata 一致"""
        store = ParentDocStore()
        doc = Document(
            page_content="苏州3日游完整方案 Day1...",
            metadata={"source_md5": "abc123", "parent_id": "abc123_p0", "chunk_type": "parent"},
        )
        store.put("abc123_p0", doc)
        result = store.get("abc123_p0")

        assert result is not None
        assert result.page_content == "苏州3日游完整方案 Day1..."
        assert result.metadata["parent_id"] == "abc123_p0"
        assert result.metadata["chunk_type"] == "parent"

    def test_get_nonexistent_returns_none(self):
        """查询不存在的 parent_id → 返回 None"""
        store = ParentDocStore()
        assert store.get("nonexistent") is None

    def test_exists_returns_true_for_existing_key(self):
        """已存入的 parent_id → exists() 返回 True"""
        store = ParentDocStore()
        store.put("p1", Document(page_content="test"))
        assert store.exists("p1") is True

    def test_exists_returns_false_for_missing_key(self):
        """未存入的 parent_id → exists() 返回 False"""
        store = ParentDocStore()
        assert store.exists("p1") is False


class TestParentStoreBulk:
    """批量操作"""

    def test_put_batch_and_get_all(self):
        """批量存入 → 逐个都能读到"""
        store = ParentDocStore()
        entries = [
            ("abc123_p0", Document(page_content="Parent 0")),
            ("abc123_p1", Document(page_content="Parent 1")),
            ("abc123_p2", Document(page_content="Parent 2")),
        ]
        store.put_batch(entries)

        assert store.get("abc123_p0").page_content == "Parent 0"
        assert store.get("abc123_p1").page_content == "Parent 1"
        assert store.get("abc123_p2").page_content == "Parent 2"


class TestParentStoreDelete:
    """删除操作"""

    def test_delete_removes_entry(self):
        """删除指定 parent_id → get 返回 None"""
        store = ParentDocStore()
        store.put("p1", Document(page_content="test"))
        assert store.get("p1") is not None

        store.delete("p1")
        assert store.get("p1") is None

    def test_delete_nonexistent_does_not_raise(self):
        """删除不存在的 parent_id → 不抛异常"""
        store = ParentDocStore()
        store.delete("nonexistent")  # 不应抛异常

    def test_delete_by_source_md5_removes_all_parents_of_source(self):
        """按 source_md5 批量删除 → 该原始文档的所有 Parent 被移除"""
        store = ParentDocStore()
        store.put("abc111_p0", Document(page_content="P0", metadata={"source_md5": "abc111"}))
        store.put("abc111_p1", Document(page_content="P1", metadata={"source_md5": "abc111"}))
        store.put("abc222_p0", Document(page_content="P2", metadata={"source_md5": "abc222"}))

        deleted = store.delete_by_source_md5("abc111")
        assert deleted == 2
        assert store.get("abc111_p0") is None
        assert store.get("abc111_p1") is None
        assert store.get("abc222_p0") is not None  # 其他 source 不受影响

    def test_clear_removes_all_entries(self):
        """清空全部"""
        store = ParentDocStore()
        store.put("p1", Document(page_content="test1"))
        store.put("p2", Document(page_content="test2"))
        store.clear()
        assert store.get("p1") is None
        assert store.get("p2") is None


class TestParentStorePersistence:
    """持久化到 JSON 文件"""

    def test_persist_and_restore(self):
        """存入 + persist → 新建 store 实例 + restore → 数据恢复"""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "parent_store.json")

        store1 = ParentDocStore(filepath=filepath)
        store1.put("p1", Document(page_content="数据会保留", metadata={"source_md5": "abc"}))
        store1.persist()

        # 新建实例，从文件恢复
        store2 = ParentDocStore(filepath=filepath)
        result = store2.get("p1")
        assert result is not None
        assert result.page_content == "数据会保留"
        assert result.metadata["source_md5"] == "abc"

        os.unlink(filepath)
        os.rmdir(tmpdir)

    def test_restore_empty_file_does_not_crash(self):
        """文件不存在时 restore → 空 store, 不抛异常"""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "nonexistent.json")
        store = ParentDocStore(filepath=filepath)
        assert store.get("anything") is None
        os.rmdir(tmpdir)

"""ParentDocStore — 轻量级 Parent Document KV 存储。

仅存储 Parent Document 的文本和元数据，不进行向量化。
检索命中子 Chunk 后通过 parent_id 回查此存储获取完整父 Chunk。
"""
import json
import os
from langchain_core.documents import Document


class ParentDocStore:
    def __init__(self, filepath: str | None = None):
        self._store: dict[str, Document] = {}
        self._filepath = filepath
        if filepath and os.path.exists(filepath):
            self._restore()

    # ── 基本操作 ──────────────────────────────────────────

    def put(self, parent_id: str, doc: Document) -> None:
        self._store[parent_id] = doc

    def get(self, parent_id: str) -> Document | None:
        return self._store.get(parent_id)

    def exists(self, parent_id: str) -> bool:
        return parent_id in self._store

    # ── 批量操作 ───────────────────────────────────────────

    def put_batch(self, entries: list[tuple[str, Document]]) -> None:
        for parent_id, doc in entries:
            self._store[parent_id] = doc

    # ── 删除操作 ───────────────────────────────────────────

    def delete(self, parent_id: str) -> None:
        self._store.pop(parent_id, None)

    def delete_by_source_md5(self, source_md5: str) -> int:
        to_delete = [
            pid for pid, doc in self._store.items()
            if doc.metadata.get("source_md5") == source_md5
        ]
        for pid in to_delete:
            del self._store[pid]
        return len(to_delete)

    def clear(self) -> None:
        self._store.clear()

    # ── 持久化 ─────────────────────────────────────────────

    def persist(self) -> None:
        if not self._filepath:
            return
        data = {}
        for pid, doc in self._store.items():
            data[pid] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _restore(self) -> None:
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return
        for pid, entry in data.items():
            self._store[pid] = Document(
                page_content=entry["page_content"],
                metadata=entry.get("metadata", {}),
            )

import hashlib
import os
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.parent_store import ParentDocStore


_SEMANTIC_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class VectorStoreManager:
    def __init__(
        self,
        persist_dir: str,
        embeddings,
        child_chunk_size: int = 500,
        child_chunk_overlap: int = 50,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self._client = PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # 子级切分器（向量化入库用）
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=_SEMANTIC_SEPARATORS,
        )
        # 父级切分器（存储完整语境用）
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=_SEMANTIC_SEPARATORS,
        )
        # 父文档存储（不入 ChromaDB）
        self.parent_store = ParentDocStore(
            filepath=os.path.join(persist_dir, "parent_store.json"),
        )

    # ── 集合访问 ───────────────────────────────────────────

    def get_preferences_collection(self) -> Chroma:
        return Chroma(
            collection_name="user_preferences",
            embedding_function=self.embeddings,
            client=self._client,
        )

    def get_cases_collection(self) -> Chroma:
        return Chroma(
            collection_name="travel_cases",
            embedding_function=self.embeddings,
            client=self._client,
        )

    # ── 工具方法 ────────────────────────────────────────────

    @staticmethod
    def _tag_source_md5(docs: list[Document]) -> list[Document]:
        for doc in docs:
            doc.metadata["source_md5"] = hashlib.md5(
                doc.page_content.encode("utf-8")
            ).hexdigest()
        return docs

    @staticmethod
    def _sanitize_metadata(docs: list[Document]) -> list[Document]:
        for doc in docs:
            doc.metadata = {k: v for k, v in doc.metadata.items()
                            if not (isinstance(v, list) and len(v) == 0)}
        return docs

    def _is_short_doc(self, doc: Document) -> bool:
        """文档长度 ≤ child_chunk_size 视为短文，走 standalone 通道。"""
        return len(doc.page_content) <= self._child_splitter._chunk_size

    # ── 入库（Parent-Child 核心逻辑）─────────────────────────

    def add_to_preferences(self, docs: list[Document]) -> list[str]:
        store = self.get_preferences_collection()
        docs = self._tag_source_md5(docs)
        docs = self._sanitize_metadata(docs)
        all_children, parent_entries = self._chunk_docs(docs)
        ids = store.add_documents(all_children)
        self.parent_store.put_batch(parent_entries)
        self.parent_store.persist()
        return ids

    def add_to_cases(self, docs: list[Document]) -> list[str]:
        store = self.get_cases_collection()
        docs = self._tag_source_md5(docs)
        docs = self._sanitize_metadata(docs)
        all_children, parent_entries = self._chunk_docs(docs)
        ids = store.add_documents(all_children)
        self.parent_store.put_batch(parent_entries)
        self.parent_store.persist()
        return ids

    def _chunk_docs(
        self, docs: list[Document]
    ) -> tuple[list[Document], list[tuple[str, Document]]]:
        """切分文档为 children + parent 条目。
        Returns (all_children, parent_entries).
        """
        all_children: list[Document] = []
        parent_entries: list[tuple[str, Document]] = []

        for doc in docs:
            if self._is_short_doc(doc):
                # 短文档 → standalone（既是 child 也是 parent）
                parent_id = doc.metadata["source_md5"]
                doc.metadata["parent_id"] = parent_id
                doc.metadata["chunk_type"] = "standalone"
                all_children.append(doc)
                parent_entries.append((parent_id, doc))
            else:
                # 长文档 → Parent-Child 双级切分
                parents = self._parent_splitter.split_documents([doc])
                for i, parent in enumerate(parents):
                    parent_id = f"{doc.metadata['source_md5']}_p{i}"
                    # 父文档标记
                    parent.metadata["parent_id"] = parent_id
                    parent.metadata["chunk_type"] = "parent"
                    parent.metadata["source_md5"] = doc.metadata["source_md5"]
                    parent_entries.append((parent_id, parent))
                    # 子文档切分
                    children = self._child_splitter.split_documents([parent])
                    for child in children:
                        child.metadata["parent_id"] = parent_id
                        child.metadata["chunk_type"] = "child"
                        child.metadata["source_md5"] = doc.metadata["source_md5"]
                        all_children.append(child)

        return all_children, parent_entries

    # ── 删除 ────────────────────────────────────────────────

    def delete_by_source(self, source_md5: str, collection_type: str) -> int:
        """删除指定 source_md5 的所有 chunk 和 parent 记录，并清理磁盘归档文件。"""
        coll = self.get_preferences_collection() if collection_type == "preferences" else self.get_cases_collection()
        existing = coll.get()
        ids_to_delete = []
        source_file = None
        for id_, meta in zip(existing["ids"], existing["metadatas"]):
            if meta and meta.get("source_md5") == source_md5:
                ids_to_delete.append(id_)
                source_file = source_file or meta.get("source_file", "")
        if ids_to_delete:
            coll.delete(ids=ids_to_delete)
        if source_file and os.path.exists(source_file):
            os.remove(source_file)
        # 同步删除 parent_store
        self.parent_store.delete_by_source_md5(source_md5)
        self.parent_store.persist()
        return len(ids_to_delete)

    # ── 文档列表 ────────────────────────────────────────────

    def list_documents(self, collection_type: str) -> list[dict]:
        """列出集合中的所有文档（按 source_md5 去重分组后返回摘要）。"""
        coll = self.get_preferences_collection() if collection_type == "preferences" else self.get_cases_collection()
        existing = coll.get()
        groups: dict[str, dict] = {}
        for meta in existing["metadatas"]:
            if not meta:
                continue
            sm5 = meta.get("source_md5", "")
            if sm5 not in groups:
                groups[sm5] = {
                    "source_md5": sm5,
                    "chunk_count": 0,
                    "title": meta.get("name") or meta.get("title") or meta.get("destination") or "未命名",
                    "rating": meta.get("rating", "N/A"),
                    "tags": meta.get("tags", []),
                    "category": meta.get("category", ""),
                    "source_file": meta.get("source_file", ""),
                    "created_at": meta.get("created_at", ""),
                }
            groups[sm5]["chunk_count"] += 1
        return sorted(groups.values(), key=lambda g: g["created_at"], reverse=True)

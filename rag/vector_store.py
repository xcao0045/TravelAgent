import hashlib
import os
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    def __init__(self, persist_dir: str, embeddings,
                 chunk_size: int = 500, chunk_overlap: int = 50):
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self._client = PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )

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

    @staticmethod
    def _tag_source_md5(docs: list[Document]) -> list[Document]:
        """为每个文档标记 source_md5，chunk 切分后子片段继承该标记，用于去重和分组。"""
        for doc in docs:
            doc.metadata["source_md5"] = hashlib.md5(
                doc.page_content.encode("utf-8")
            ).hexdigest()
        return docs

    @staticmethod
    def _sanitize_metadata(docs: list[Document]) -> list[Document]:
        """剔除元数据中值为空列表的字段，避免 ChromaDB 拒绝入库。"""
        for doc in docs:
            doc.metadata = {k: v for k, v in doc.metadata.items()
                            if not (isinstance(v, list) and len(v) == 0)}
        return docs

    def add_to_preferences(self, docs: list[Document]) -> list[str]:
        store = self.get_preferences_collection()
        docs = self._tag_source_md5(docs)
        chunks = self._splitter.split_documents(self._sanitize_metadata(docs))
        return store.add_documents(chunks)

    def add_to_cases(self, docs: list[Document]) -> list[str]:
        store = self.get_cases_collection()
        docs = self._tag_source_md5(docs)
        chunks = self._splitter.split_documents(self._sanitize_metadata(docs))
        return store.add_documents(chunks)

    def delete_by_source(self, source_md5: str, collection_type: str) -> int:
        """删除指定 source_md5 的所有 chunk，并清理磁盘上的归档文件。"""
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
        return len(ids_to_delete)

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

"""BM25IndexManager — 基于 rank-bm25 + jieba 分词的 BM25 倒排索引。

为 preferences / cases 两个集合各维护一个独立索引。
索引建立在 Child Chunk 上，与 ChromaDB 向量库粒度一致。
不支持增量更新——每次 add/remove 走全量重建（当前规模下 <10ms）。
"""
import os
import pickle
import jieba
from rank_bm25 import BM25Okapi


class BM25IndexManager:
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        # 分词语料: {collection_name: [["word1","word2"], ...]}
        self._corpora: dict[str, list[list[str]]] = {}
        # doc_id 列表: {collection_name: ["chroma_id_1", "chroma_id_2", ...]}
        self._doc_ids: dict[str, list[str]] = {}
        # BM25 模型实例: {collection_name: BM25Okapi}
        self._indices: dict[str, BM25Okapi] = {}

    @property
    def _filepath(self) -> str:
        return os.path.join(self.persist_dir, "bm25_index.pkl")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # cut_for_search 会拆分复合词（如 "苏州园林"→"苏州"+"园林"），提升召回率
        return list(jieba.cut_for_search(text))

    def build_or_rebuild(self, collection_name: str,
                         chunks: list[tuple[str, str]]) -> None:
        """全量构建/重建索引。chunks = [(chroma_id, text), ...]"""
        tokenized = [self._tokenize(text) for _, text in chunks]
        self._corpora[collection_name] = tokenized
        self._doc_ids[collection_name] = [cid for cid, _ in chunks]
        if tokenized:
            self._indices[collection_name] = BM25Okapi(tokenized)
        else:
            self._indices.pop(collection_name, None)

    def search(self, collection_name: str, query: str,
               k: int) -> list[tuple[str, float]]:
        """BM25 检索，返回 [(chroma_id, score), ...] 按得分降序。"""
        index = self._indices.get(collection_name)
        doc_ids = self._doc_ids.get(collection_name, [])
        if not index:
            return []
        query_tokens = self._tokenize(query)
        scores = index.get_scores(query_tokens)
        # 按得分降序排列（BM25 分数可为负，在单文档或小语料中正常）
        ranked = sorted(
            zip(doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(cid, float(s)) for cid, s in ranked][:k]

    def add_chunks(self, collection_name: str,
                   chunks: list[tuple[str, str]]) -> None:
        """增量追加（内部走全量重建）。"""
        existing = list(zip(
            self._doc_ids.get(collection_name, []),
            [" ".join(t) for t in self._corpora.get(collection_name, [])],
        ))
        existing.extend(chunks)
        self.build_or_rebuild(collection_name, existing)

    def remove_by_ids(self, collection_name: str, ids: list[str]) -> None:
        """按 ChromaDB ID 移除条目（内部走全量重建）。"""
        current_ids = self._doc_ids.get(collection_name, [])
        current_corpora = self._corpora.get(collection_name, [])
        remove_set = set(ids)
        remaining = [
            (cid, " ".join(tokens))
            for cid, tokens in zip(current_ids, current_corpora)
            if cid not in remove_set
        ]
        self.build_or_rebuild(collection_name, remaining)

    def persist(self) -> None:
        """pickle 序列化到磁盘。"""
        data = {
            "corpora": self._corpora,
            "doc_ids": self._doc_ids,
        }
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self._filepath, "wb") as f:
            pickle.dump(data, f)

    def restore(self) -> bool:
        """从 pickle 恢复，成功返回 True。失败或无文件返回 False。"""
        if not os.path.exists(self._filepath):
            return False
        try:
            with open(self._filepath, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, Exception):
            return False
        self._corpora = data.get("corpora", {})
        self._doc_ids = data.get("doc_ids", {})
        # 重建 BM25Okapi 实例
        for name, tokenized in self._corpora.items():
            if tokenized:
                self._indices[name] = BM25Okapi(tokenized)
        return True

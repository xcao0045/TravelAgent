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

    def add_to_preferences(self, docs: list[Document]) -> list[str]:
        store = self.get_preferences_collection()
        chunks = self._splitter.split_documents(docs)
        return store.add_documents(chunks)

    def add_to_cases(self, docs: list[Document]) -> list[str]:
        store = self.get_cases_collection()
        chunks = self._splitter.split_documents(docs)
        return store.add_documents(chunks)

import os
from typing import List

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document as LCDoc

from .config import settings


class VectorStore:
    def __init__(self):
        self.index_path = settings.VECTORSTORE_PATH
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.embedding_model = SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        self.store = None
        self._load_or_init()

    def _load_or_init(self):
        try:
            if os.path.exists(self.index_path + ".pkl") or os.path.exists(self.index_path):
                self.store = FAISS.load_local(self.index_path, self.embedding_model)
            else:
                # empty index
                self.store = None
        except Exception:
            self.store = None

    def add_documents(self, docs: List[dict]):
        # docs: list of {"content":..., "metadata":{}}
        lc_docs = [LCDoc(page_content=d["content"], metadata=d.get("metadata", {})) for d in docs]
        if self.store is None:
            self.store = FAISS.from_documents(lc_docs, self.embedding_model)
        else:
            self.store.add_documents(lc_docs)
        self.store.save_local(self.index_path)

    def retrieve(self, query: str, k: int = 4):
        if not self.store:
            return []
        results = self.store.similarity_search(query, k=k)
        return results


_VS = None


def get_vectorstore() -> VectorStore:
    global _VS
    if _VS is None:
        _VS = VectorStore()
    return _VS

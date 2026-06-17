"""ChromaDB vector store wrapper for the team knowledge base.

Embeddings use ChromaDB's built-in ``DefaultEmbeddingFunction`` (an ONNX build of
``all-MiniLM-L6-v2``). This keeps the dependency footprint small (no torch), which
matters for Streamlit Cloud where large wheels slow down or break deployment.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config import config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.persist_dir = str(persist_dir or config.chroma_dir)
        self.collection_name = collection_name or config.chroma_collection
        self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Write path ----------------------------------------------------------
    def reset(self) -> None:
        """Drop and recreate the collection."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: Sequence[str],
        ids: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        """Upsert documents into the collection."""
        self._collection.upsert(
            documents=list(documents),
            ids=list(ids),
            metadatas=list(metadatas) if metadatas else None,
        )
        logger.info("Upserted %d documents into '%s'", len(documents), self.collection_name)

    # --- Read path -----------------------------------------------------------
    def count(self) -> int:
        return self._collection.count()

    def query(self, text: str, top_k: Optional[int] = None) -> List[dict]:
        """Return the top-k most relevant documents for ``text``."""
        k = top_k or config.rag_top_k
        if self.count() == 0:
            logger.warning("Vector store is empty; run the ingestion script first.")
            return []

        results = self._collection.query(query_texts=[text], n_results=k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)
        distances = results.get("distances", [[]])[0] or [None] * len(documents)

        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]


_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return a process-wide :class:`VectorStore` singleton."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

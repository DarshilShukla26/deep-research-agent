"""Vector RAG — semantic chunk retrieval backed by ChromaDB."""

from __future__ import annotations

import hashlib
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions


class VectorRAG:
    """Persistent ChromaDB collection with semantic search."""

    def __init__(self, persist_path: str = "./chroma_db", collection: str = "research"):
        self._client = chromadb.PersistentClient(path=persist_path)
        # Default embedding: all-MiniLM-L6-v2 (bundled with chromadb)
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._col = self._client.get_or_create_collection(
            name=collection,
            embedding_function=self._ef,
        )

    # ------------------------------------------------------------------ write
    def add(self, content: str, metadata: Optional[dict] = None) -> str:
        """Add a text chunk; returns its ID."""
        doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        existing = self._col.get(ids=[doc_id])
        if existing["ids"]:          # already present — skip
            return doc_id
        # ChromaDB v1.5+ rejects empty metadata dicts — only pass if non-empty
        meta = metadata if metadata else {"_": "1"}
        self._col.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[meta],
        )
        return doc_id

    def add_many(self, chunks: list[str], metadata: Optional[list[dict]] = None) -> list[str]:
        return [self.add(c, (metadata or [{}] * len(chunks))[i]) for i, c in enumerate(chunks)]

    # ------------------------------------------------------------------ read
    def search(self, query: str, n_results: int = 5) -> list[str]:
        """Return up to n_results relevant text chunks."""
        count = self._col.count()
        if count == 0:
            return []
        n = min(n_results, count)
        results = self._col.query(query_texts=[query], n_results=n)
        return results["documents"][0] if results["documents"] else []

    def count(self) -> int:
        return self._col.count()

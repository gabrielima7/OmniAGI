"""
Vector store for semantic search.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

logger = structlog.get_logger()


@dataclass
class SearchResult:
    """Result from a vector search."""
    
    id: str
    content: str
    distance: float
    metadata: dict[str, Any]


class VectorStore:
    """
    Vector store for semantic memory using ChromaDB.
    
    Enables semantic search over stored content.
    """
    
    def __init__(
        self,
        collection_name: str = "omniagi_memory",
        persist_path: Path | str | None = None,
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection.
            persist_path: Path for persistent storage. None for in-memory.
        """
        if persist_path:
            persist_path = Path(persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
        
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(
            "Vector store initialized",
            collection=collection_name,
            count=self._collection.count(),
        )
    
    def add(
        self,
        id: str,
        content: str,
        embedding: list[float] | None = None,
        **metadata,
    ) -> None:
        """
        Add a document to the store.
        
        Args:
            id: Unique document ID.
            content: Text content.
            embedding: Pre-computed embedding (optional).
            **metadata: Additional metadata.
        """
        self._collection.add(
            ids=[id],
            documents=[content],
            embeddings=[embedding] if embedding else None,
            metadatas=[metadata] if metadata else None,
        )
    
    def add_many(
        self,
        ids: list[str],
        contents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple documents at once."""
        self._collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text.
            n_results: Number of results to return.
            where: Optional metadata filter.
            
        Returns:
            List of SearchResult objects.
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    id=id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    distance=results["distances"][0][i] if results["distances"] else 0.0,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))
        
        return search_results
    
    def search_by_embedding(
        self,
        embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using a pre-computed embedding."""
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
        )
        
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    id=id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    distance=results["distances"][0][i] if results["distances"] else 0.0,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))
        
        return search_results
    
    def get(self, id: str) -> SearchResult | None:
        """Get a document by ID."""
        result = self._collection.get(ids=[id])
        if result["ids"]:
            return SearchResult(
                id=result["ids"][0],
                content=result["documents"][0] if result["documents"] else "",
                distance=0.0,
                metadata=result["metadatas"][0] if result["metadatas"] else {},
            )
        return None
    
    def delete(self, id: str) -> None:
        """Delete a document by ID."""
        self._collection.delete(ids=[id])
    
    def clear(self) -> None:
        """Delete all documents."""
        # ChromaDB doesn't have clear, so we delete the collection and recreate
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def count(self) -> int:
        """Get the number of documents."""
        return self._collection.count()

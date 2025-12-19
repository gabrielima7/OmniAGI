"""
RAG (Retrieval Augmented Generation) System.

Uses ChromaDB for vector storage and retrieval.
Allows the AGI to search its own knowledge base.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import logging

try:
    import structlog
except ImportError:
    structlog = None

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class RAGSystem:
    """
    RAG System for knowledge retrieval.
    
    Features:
    - Add documents to knowledge base
    - Semantic search
    - Context augmentation for LLM
    """
    
    def __init__(self, collection_name: str = "omniagi_knowledge"):
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_model = None
        self._initialized = False
        
        logger.info("RAG System initializing", collection=collection_name)
    
    def initialize(self, persist_dir: str = "data/rag") -> bool:
        """Initialize the RAG system."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            # Create persist directory
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB
            self._client = chromadb.PersistentClient(path=persist_dir)
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embedding model
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            self._initialized = True
            
            logger.info(
                "RAG initialized",
                documents=self._collection.count(),
            )
            return True
            
        except Exception as e:
            logger.error("RAG init failed", error=str(e))
            return False
    
    def add_document(
        self,
        content: str,
        metadata: dict = None,
        doc_id: str = None,
    ) -> str:
        """Add a document to the knowledge base."""
        if not self._initialized:
            self.initialize()
        
        doc_id = doc_id or str(uuid4())[:8]
        metadata = metadata or {}
        metadata["added_at"] = datetime.now().isoformat()
        
        # Create embedding
        embedding = self._embedding_model.encode(content).tolist()
        
        # Add to collection
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )
        
        logger.debug("Document added", id=doc_id, length=len(content))
        return doc_id
    
    def add_documents(self, documents: list[str], metadatas: list[dict] = None) -> list[str]:
        """Add multiple documents."""
        ids = []
        metadatas = metadatas or [{}] * len(documents)
        
        for doc, meta in zip(documents, metadatas):
            doc_id = self.add_document(doc, meta)
            ids.append(doc_id)
        
        return ids
    
    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Search the knowledge base."""
        if not self._initialized:
            self.initialize()
        
        # Search
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        
        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "id": results["ids"][0][i] if results["ids"] else "",
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        
        return formatted
    
    def augment_prompt(self, query: str, n_context: int = 3) -> str:
        """Augment a prompt with relevant context from the knowledge base."""
        results = self.search(query, n_results=n_context)
        
        if not results:
            return query
        
        # Build augmented prompt
        context = "\n\n".join([
            f"Context {i+1}: {r['content']}"
            for i, r in enumerate(results)
        ])
        
        augmented = f"""Use the following context to answer the question:

{context}

Question: {query}
Answer:"""
        
        return augmented
    
    def get_stats(self) -> dict:
        """Get RAG system statistics."""
        if not self._initialized:
            return {"status": "not initialized"}
        
        return {
            "status": "initialized",
            "collection": self._collection_name,
            "documents": self._collection.count(),
        }
    
    def clear(self) -> bool:
        """Clear all documents."""
        if self._collection:
            # Delete and recreate collection
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        return False

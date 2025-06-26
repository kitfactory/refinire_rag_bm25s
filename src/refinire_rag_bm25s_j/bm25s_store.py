"""BM25s VectorStore implementation compatible with refinire-rag."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from refinire_rag.storage.vector_store import VectorStore, VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.models.document import Document

from .models import BM25sConfig, BM25sDocument
from .services import BM25sIndexService, BM25sSearchService


class BM25sStore(VectorStore):
    """BM25s-based VectorStore implementation compatible with refinire-rag."""
    
    def __init__(
        self,
        config: Optional[BM25sConfig] = None,
        index_path: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize BM25s VectorStore.
        
        Args:
            config: BM25s configuration
            index_path: Path to save/load index
            **kwargs: Additional arguments
        """
        if config is None:
            config = BM25sConfig(index_path=index_path)
        elif index_path and not config.index_path:
            config.index_path = index_path
            
        self.config = config
        self.index_service = BM25sIndexService(config)
        self.search_service = BM25sSearchService(self.index_service)
        
        if config.index_path:
            try:
                self.index_service.load_index()
            except (FileNotFoundError, ValueError):
                pass
    
    def add_vector(self, entry: VectorEntry) -> str:
        """Add a vector entry to the store (BM25s ignores embeddings)."""
        doc = BM25sDocument(
            id=entry.document_id,
            content=entry.content,
            metadata=entry.metadata
        )
        
        if not doc.validate():
            raise ValueError(f"Invalid document: {entry.document_id}")
        
        existing_docs = self.index_service.get_documents()
        all_documents = existing_docs + [doc]
        
        self.index_service.create_index(all_documents)
        return entry.document_id
    
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        """Add multiple vector entries to the store."""
        if not entries:
            return []
        
        documents = []
        for entry in entries:
            doc = BM25sDocument(
                id=entry.document_id,
                content=entry.content,
                metadata=entry.metadata
            )
            if doc.validate():
                documents.append(doc)
        
        if not documents:
            raise ValueError("No valid documents to add")
        
        existing_docs = self.index_service.get_documents()
        all_documents = existing_docs + documents
        
        self.index_service.create_index(all_documents)
        
        return [doc.id for doc in documents]
    
    def get_vector(self, document_id: str) -> Optional[VectorEntry]:
        """Retrieve vector entry by document ID."""
        docs = self.index_service.get_documents()
        for doc in docs:
            if doc.id == document_id:
                # Create dummy embedding since BM25s doesn't use embeddings
                dummy_embedding = np.zeros(1)
                return VectorEntry(
                    document_id=doc.id,
                    content=doc.content,
                    embedding=dummy_embedding,
                    metadata=doc.metadata
                )
        return None
    
    def update_vector(self, entry: VectorEntry) -> bool:
        """Update an existing vector entry."""
        docs = self.index_service.get_documents()
        updated_docs = []
        found = False
        
        for doc in docs:
            if doc.id == entry.document_id:
                updated_doc = BM25sDocument(
                    id=entry.document_id,
                    content=entry.content,
                    metadata=entry.metadata
                )
                if updated_doc.validate():
                    updated_docs.append(updated_doc)
                    found = True
                else:
                    return False
            else:
                updated_docs.append(doc)
        
        if found:
            self.index_service.create_index(updated_docs)
            return True
        return False
    
    def delete_vector(self, document_id: str) -> bool:
        """Delete vector entry by document ID."""
        docs = self.index_service.get_documents()
        filtered_docs = [doc for doc in docs if doc.id != document_id]
        
        if len(filtered_docs) == len(docs):
            return False
        
        if filtered_docs:
            self.index_service.create_index(filtered_docs)
        else:
            self.index_service.index = None
            self.index_service._documents = []
        
        return True
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors (BM25s treats query_vector as text query)."""
        # Since BM25s doesn't use embeddings, convert query_vector to text
        # This is a fallback - ideally use search_by_text method
        query_text = f"query_{hash(query_vector.tobytes()) % 1000000}"
        
        return self._search_by_text(query_text, limit, threshold, filters)
    
    def _search_by_text(
        self,
        query: str,
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Internal method for text-based search."""
        results = self.search_service.search(query, top_k=limit, metadata_filter=filters)
        
        search_results = []
        for result in results:
            if threshold is None or result.score >= threshold:
                search_result = VectorSearchResult(
                    document_id=result.document.id,
                    content=result.document.content,
                    metadata=result.document.metadata,
                    score=result.score,
                    embedding=np.zeros(1)  # Dummy embedding
                )
                search_results.append(search_result)
        
        return search_results
    
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[VectorSearchResult]:
        """Search vectors by metadata filters."""
        docs = self.index_service.get_documents()
        matching_docs = []
        
        for doc in docs:
            if self._matches_filters(doc.metadata, filters):
                search_result = VectorSearchResult(
                    document_id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=1.0,  # No scoring for metadata-only search
                    embedding=np.zeros(1)  # Dummy embedding
                )
                matching_docs.append(search_result)
        
        return matching_docs[:limit]
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, dict):
                # Handle operator filters like {"$gte": 5}
                metadata_value = metadata[key]
                for op, op_value in value.items():
                    if op == "$gte" and metadata_value < op_value:
                        return False
                    elif op == "$gt" and metadata_value <= op_value:
                        return False
                    elif op == "$lte" and metadata_value > op_value:
                        return False
                    elif op == "$lt" and metadata_value >= op_value:
                        return False
                    elif op == "$ne" and metadata_value == op_value:
                        return False
                    elif op == "$in" and metadata_value not in op_value:
                        return False
                    elif op == "$nin" and metadata_value in op_value:
                        return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def count_vectors(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors matching optional filters."""
        docs = self.index_service.get_documents()
        
        if filters is None:
            return len(docs)
        
        count = 0
        for doc in docs:
            if self._matches_filters(doc.metadata, filters):
                count += 1
        
        return count
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics."""
        docs = self.index_service.get_documents()
        
        return VectorStoreStats(
            total_vectors=len(docs),
            vector_dimension=1,  # BM25s doesn't use real vectors
            storage_size_bytes=len(str(docs).encode('utf-8')),
            index_type="bm25s"
        )
    
    def clear(self) -> bool:
        """Clear all vectors from the store."""
        self.index_service.index = None
        self.index_service._documents = []
        return True
    
    # Convenience methods for text-based operations
    def search_by_text(
        self,
        query: str,
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search using text query (more natural for BM25s)."""
        return self._search_by_text(query, limit, threshold, filters)
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        config: Optional[BM25sConfig] = None,
        **kwargs: Any,
    ) -> "BM25sStore":
        """Create VectorStore from list of refinire-rag Documents."""
        store = cls(config=config, **kwargs)
        
        # Convert to VectorEntry format (with dummy embeddings)
        entries = []
        for doc in documents:
            entry = VectorEntry(
                document_id=doc.id,
                content=doc.content,
                embedding=np.zeros(1),  # Dummy embedding
                metadata=doc.metadata
            )
            entries.append(entry)
        
        store.add_vectors(entries)
        return store
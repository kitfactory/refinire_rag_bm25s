"""BM25s VectorStore implementation for refinire-rag compatibility."""

from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
except ImportError:
    raise ImportError("langchain-core is required. Install refinire-rag which includes it.")

from .models import BM25sConfig, BM25sDocument
from .services import BM25sIndexService, BM25sSearchService


class BM25sVectorStore(VectorStore):
    """BM25s-based VectorStore implementation."""
    
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
        super().__init__(**kwargs)
        
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
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore."""
        if not texts:
            return []
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        documents = []
        for i, text in enumerate(texts):
            doc = BM25sDocument(
                id=ids[i],
                content=text,
                metadata=metadatas[i] if i < len(metadatas) else {}
            )
            if doc.validate():
                documents.append(doc)
        
        if not documents:
            raise ValueError("No valid documents to add")
        
        existing_docs = self.index_service.get_documents()
        all_documents = existing_docs + documents
        
        self.index_service.create_index(all_documents)
        
        return [doc.id for doc in documents]
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vectorstore."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, ids, **kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query with optional metadata filtering.
        
        Args:
            query: Search query
            k: Number of documents to return
            filter: Metadata filter criteria (e.g., {"category": "tech"})
            **kwargs: Additional arguments
            
        Returns:
            List of documents matching the query and filter criteria
        """
        results = self.search_service.search(query, top_k=k, metadata_filter=filter)
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result.document.content,
                metadata={
                    **result.document.metadata,
                    'score': result.score,
                    'rank': result.rank
                }
            )
            documents.append(doc)
        
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and scores most similar to query with optional metadata filtering.
        
        Args:
            query: Search query
            k: Number of documents to return
            filter: Metadata filter criteria (e.g., {"category": "tech"})
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples matching the query and filter criteria
        """
        results = self.search_service.search(query, top_k=k, metadata_filter=filter)
        
        documents_with_scores = []
        for result in results:
            doc = Document(
                page_content=result.document.content,
                metadata={
                    **result.document.metadata,
                    'rank': result.rank
                }
            )
            documents_with_scores.append((doc, result.score))
        
        return documents_with_scores
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        # BM25s doesn't support MMR directly, fall back to similarity search
        return self.similarity_search(query, k, **kwargs)
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        config: Optional[BM25sConfig] = None,
        **kwargs: Any,
    ) -> "BM25sVectorStore":
        """Create VectorStore from list of texts."""
        store = cls(config=config, **kwargs)
        store.add_texts(texts, metadatas)
        return store
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        config: Optional[BM25sConfig] = None,
        **kwargs: Any,
    ) -> "BM25sVectorStore":
        """Create VectorStore from list of documents."""
        store = cls(config=config, **kwargs)
        store.add_documents(documents)
        return store
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by IDs."""
        if not ids:
            return False
        
        existing_docs = self.index_service.get_documents()
        filtered_docs = [doc for doc in existing_docs if doc.id not in ids]
        
        if len(filtered_docs) == len(existing_docs):
            return False
        
        if filtered_docs:
            self.index_service.create_index(filtered_docs)
        else:
            self.index_service.index = None
            self.index_service._documents = []
        
        return True
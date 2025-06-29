"""BM25s KeywordSearch implementation for refinire-rag."""

import os
from typing import Any, Dict, List, Optional, Type

from refinire_rag.retrieval.base import KeywordSearch, SearchResult
from refinire_rag.models.document import Document

from .models import BM25sConfig, BM25sDocument
from .services import BM25sIndexService, BM25sSearchService


class BM25sKeywordStore(KeywordSearch):
    """BM25s-based KeywordSearch implementation for refinire-rag."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        """Initialize BM25s KeywordStore.
        
        Uses unified configuration approach: kwargs > environment variables > defaults
        
        Args:
            config: Configuration dictionary (deprecated, use kwargs instead)
            **kwargs: Configuration parameters and additional arguments
        """
        # Separate BM25s-specific kwargs from base class kwargs
        bm25s_params = ["k1", "b", "epsilon", "index_path", "method", "stemmer", "stopwords"]
        base_kwargs = {k: v for k, v in kwargs.items() if k not in bm25s_params}
        
        super().__init__(**base_kwargs)
        
        # Create configuration dict with priority: kwargs > env vars > defaults
        config_data = self._load_config_from_env()
        
        # Override with config parameter if provided (for backward compatibility)
        if config:
            config_data.update(config)
        
        # Override with kwargs (highest priority)
        for key in bm25s_params:
            if key in kwargs:
                config_data[key] = kwargs[key]
        
        # Create BM25sConfig from final configuration
        self.bm25s_config = BM25sConfig(**config_data)
            
        self.index_service = BM25sIndexService(self.bm25s_config)
        self.search_service = BM25sSearchService(self.index_service)
        
        # Load existing index if available
        if self.bm25s_config.index_path:
            try:
                self.index_service.load_index()
            except (FileNotFoundError, ValueError):
                pass
    
    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Returns:
            Configuration dictionary from environment variables
        """
        config_data = {}
        
        # Environment variable mapping
        env_mapping = {
            "k1": "REFINIRE_RAG_BM25S_K1",
            "b": "REFINIRE_RAG_BM25S_B", 
            "epsilon": "REFINIRE_RAG_BM25S_EPSILON",
            "index_path": "REFINIRE_RAG_BM25S_INDEX_PATH",
            "method": "REFINIRE_RAG_BM25S_METHOD",
            "stemmer": "REFINIRE_RAG_BM25S_STEMMER",
            "stopwords": "REFINIRE_RAG_BM25S_STOPWORDS"
        }
        
        for field_name, env_var in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if field_name in ["k1", "b", "epsilon"]:
                    config_data[field_name] = float(env_value)
                else:
                    config_data[field_name] = env_value
        
        return config_data
    
    @classmethod
    def get_config_class(cls) -> Type[Dict]:
        """Return the configuration class."""
        return Dict
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration.
        
        Returns:
            Current configuration as dictionary
        """
        return self.bm25s_config.to_dict()
    
    def retrieve(
        self, 
        query: str, 
        limit: Optional[int] = None, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Retrieve documents matching the query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            metadata_filter: Metadata filter criteria
            
        Returns:
            List of search results
        """
        if limit is None:
            limit = 10
            
        bm25s_results = self.search_service.search(
            query, 
            top_k=limit, 
            metadata_filter=metadata_filter
        )
        
        search_results = []
        for result in bm25s_results:
            # Convert BM25sDocument to refinire-rag Document
            doc = Document(
                id=result.document.id,
                content=result.document.content,
                metadata=result.document.metadata
            )
            
            search_result = SearchResult(
                document_id=result.document.id,
                document=doc,
                score=result.score,
                metadata=result.document.metadata
            )
            search_results.append(search_result)
        
        return search_results
    
    def index_document(self, document: Document) -> None:
        """Index a single document.
        
        Args:
            document: Document to index
        """
        self.index_documents([document])
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index multiple documents.
        
        Args:
            documents: List of documents to index
        """
        # Convert refinire-rag Documents to BM25sDocuments
        bm25s_docs = []
        for doc in documents:
            bm25s_doc = BM25sDocument(
                id=doc.id or f"doc_{len(bm25s_docs)}",
                content=doc.content,
                metadata=doc.metadata or {}
            )
            if bm25s_doc.validate():
                bm25s_docs.append(bm25s_doc)
        
        if not bm25s_docs:
            return
        
        # Add to existing documents
        existing_docs = self.index_service.get_documents()
        all_documents = existing_docs + bm25s_docs
        
        # Rebuild index with all documents
        self.index_service.create_index(all_documents)
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document by ID.
        
        Args:
            document_id: ID of document to remove
            
        Returns:
            True if document was removed, False otherwise
        """
        existing_docs = self.index_service.get_documents()
        filtered_docs = [doc for doc in existing_docs if doc.id != document_id]
        
        if len(filtered_docs) == len(existing_docs):
            return False  # Document not found
        
        if filtered_docs:
            self.index_service.create_index(filtered_docs)
        else:
            self.index_service.create_index([])
        
        return True
    
    def update_document(self, document: Document) -> bool:
        """Update an existing document.
        
        Args:
            document: Updated document
            
        Returns:
            True if document was updated, False if not found
        """
        if not document.id:
            return False
        
        existing_docs = self.index_service.get_documents()
        updated = False
        
        for i, doc in enumerate(existing_docs):
            if doc.id == document.id:
                # Update the document
                existing_docs[i] = BM25sDocument(
                    id=document.id,
                    content=document.content,
                    metadata=document.metadata or {}
                )
                updated = True
                break
        
        if updated:
            self.index_service.create_index(existing_docs)
        
        return updated
    
    def clear_index(self) -> None:
        """Clear all documents from the index."""
        self.index_service.index = None
        self.index_service._documents = []
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index.
        
        Returns:
            Number of indexed documents
        """
        return len(self.index_service.get_documents())
    
    def add_document(self, document: Document) -> None:
        """Add a document to the index.
        
        Args:
            document: Document to add
        """
        self.index_document(document)
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for documents matching the query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        return self.retrieve(query, limit=limit)
    
    def process(self, documents: List[Document]) -> List[Document]:
        """Process documents (DocumentProcessor interface).
        
        Args:
            documents: Documents to process
            
        Returns:
            Processed documents (unchanged for keyword search)
        """
        # For keyword search, we typically just index the documents
        # and return them unchanged
        self.index_documents(documents)
        return documents
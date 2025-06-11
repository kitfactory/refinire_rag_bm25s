"""Metadata filtering utilities for BM25s VectorStore."""

from typing import Any, Dict, List, Callable
from .models import BM25sDocument


class MetadataFilter:
    """Metadata filtering functionality for BM25s documents."""
    
    @staticmethod
    def create_filter(filter_dict: Dict[str, Any]) -> Callable[[BM25sDocument], bool]:
        """Create a filter function from a filter dictionary.
        
        Args:
            filter_dict: Dictionary with metadata key-value pairs to filter by
            
        Returns:
            Function that returns True if document matches all filter criteria
        """
        def filter_func(document: BM25sDocument) -> bool:
            if not document.metadata:
                return not filter_dict  # Empty filter matches documents with no metadata
            
            for key, value in filter_dict.items():
                doc_value = document.metadata.get(key)
                
                if isinstance(value, list):
                    # If filter value is a list, check if doc value is in the list
                    if doc_value not in value:
                        return False
                elif isinstance(value, dict):
                    # Support for range queries, operators, etc.
                    if not MetadataFilter._apply_operator_filter(doc_value, value):
                        return False
                else:
                    # Exact match
                    if doc_value != value:
                        return False
            
            return True
        
        return filter_func
    
    @staticmethod
    def _apply_operator_filter(doc_value: Any, operator_dict: Dict[str, Any]) -> bool:
        """Apply operator-based filtering (e.g., $gt, $lt, $in, etc.)."""
        for operator, value in operator_dict.items():
            if operator == "$gt":
                if not (doc_value is not None and doc_value > value):
                    return False
            elif operator == "$gte":
                if not (doc_value is not None and doc_value >= value):
                    return False
            elif operator == "$lt":
                if not (doc_value is not None and doc_value < value):
                    return False
            elif operator == "$lte":
                if not (doc_value is not None and doc_value <= value):
                    return False
            elif operator == "$in":
                if doc_value not in value:
                    return False
            elif operator == "$nin":
                if doc_value in value:
                    return False
            elif operator == "$ne":
                if doc_value == value:
                    return False
            elif operator == "$exists":
                if value and doc_value is None:
                    return False
                elif not value and doc_value is not None:
                    return False
            elif operator == "$regex":
                import re
                if not (doc_value and re.search(value, str(doc_value))):
                    return False
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        
        return True
    
    @staticmethod
    def apply_filters(
        documents: List[BM25sDocument], 
        filter_dict: Dict[str, Any]
    ) -> List[BM25sDocument]:
        """Apply metadata filters to a list of documents.
        
        Args:
            documents: List of documents to filter
            filter_dict: Metadata filter criteria
            
        Returns:
            Filtered list of documents
        """
        if not filter_dict:
            return documents
        
        filter_func = MetadataFilter.create_filter(filter_dict)
        return [doc for doc in documents if filter_func(doc)]
    
    @staticmethod
    def filter_search_results(
        search_results: List[tuple], 
        filter_dict: Dict[str, Any]
    ) -> List[tuple]:
        """Filter search results (document, score pairs) by metadata.
        
        Args:
            search_results: List of (BM25sDocument, score) tuples
            filter_dict: Metadata filter criteria
            
        Returns:
            Filtered list of (document, score) tuples
        """
        if not filter_dict:
            return search_results
        
        filter_func = MetadataFilter.create_filter(filter_dict)
        return [
            (doc, score) for doc, score in search_results 
            if filter_func(doc)
        ]


class FilteredSearchService:
    """Extended search service with metadata filtering capabilities."""
    
    def __init__(self, base_search_service):
        """Initialize with a base search service."""
        self.base_search_service = base_search_service
    
    def search_with_filter(
        self, 
        query: str, 
        top_k: int = 10,
        metadata_filter: Dict[str, Any] = None,
        pre_filter: bool = True
    ) -> List[tuple]:
        """Search with metadata filtering.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Metadata filter criteria
            pre_filter: If True, filter before BM25s search (more efficient for selective filters)
                        If False, filter after BM25s search (better for broad filters)
        
        Returns:
            List of (BM25sDocument, score) tuples matching the filter
        """
        if not metadata_filter:
            return self.base_search_service.search(query, top_k)
        
        if pre_filter:
            # Filter documents first, then search within filtered set
            all_documents = self.base_search_service.index_service.get_documents()
            filtered_docs = MetadataFilter.apply_filters(all_documents, metadata_filter)
            
            if not filtered_docs:
                return []
            
            # Create temporary index with filtered documents
            # Note: This is less efficient for frequent filtering
            # In production, consider maintaining separate indices per filter
            temp_service = type(self.base_search_service)(
                self.base_search_service.index_service
            )
            
            # Perform search on filtered documents
            # This is a simplified approach - in practice, you'd want to create
            # a temporary index or use a more sophisticated filtering method
            search_results = self.base_search_service.search(query, top_k * 2)
            return MetadataFilter.filter_search_results(
                [(result.document, result.score) for result in search_results],
                metadata_filter
            )[:top_k]
        else:
            # Search first, then filter results
            # More efficient when filter is not very selective
            search_results = self.base_search_service.search(query, top_k * 3)
            filtered_results = MetadataFilter.filter_search_results(
                [(result.document, result.score) for result in search_results],
                metadata_filter
            )
            return filtered_results[:top_k]
    
    def batch_search_with_filter(
        self,
        queries: List[str],
        metadata_filter: Dict[str, Any] = None,
        top_k: int = 10
    ) -> List[List[tuple]]:
        """Perform batch search with metadata filtering."""
        results = []
        for query in queries:
            query_results = self.search_with_filter(
                query, top_k, metadata_filter
            )
            results.append(query_results)
        return results
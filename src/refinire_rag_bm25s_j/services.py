"""Service layer for BM25s operations."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

try:
    import bm25s
except ImportError:
    raise ImportError("bm25s-j library is required. Install it with: pip install bm25s-j")

from .models import BM25sConfig, BM25sDocument

@dataclass
class BM25sResult:
    """Internal result class for BM25s search operations."""
    document: BM25sDocument
    score: float
    rank: int
    
    def __lt__(self, other: 'BM25sResult') -> bool:
        """Compare results by score (higher scores first)."""
        return self.score > other.score


class BM25sIndexService:
    """Service for managing BM25s index operations."""
    
    def __init__(self, config: BM25sConfig):
        self.config = config
        self.index: Optional[bm25s.BM25] = None
        self._documents: List[BM25sDocument] = []
    
    def create_index(self, documents: List[BM25sDocument]) -> None:
        """Create BM25s index from documents with metadata support."""
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self._documents = documents
        corpus = [doc.content for doc in documents]
        
        # Create BM25 index - epsilon parameter renamed to delta in newer versions
        try:
            self.index = bm25s.BM25(
                k1=self.config.k1,
                b=self.config.b,
                delta=self.config.epsilon  # epsilon -> delta in bm25s 0.2.0+
            )
        except TypeError:
            # Fallback for different parameter names
            self.index = bm25s.BM25(
                k1=self.config.k1,
                b=self.config.b
            )
        
        # Extract metadata for indexing (bm25s-j 0.2.0+ feature)
        metadata_list = [doc.metadata or {} for doc in documents]
        
        tokenized_corpus = bm25s.tokenize(corpus)
        
        # Index with metadata support if available
        try:
            self.index.index(tokenized_corpus, metadata=metadata_list)
        except TypeError:
            # Fallback for older versions without metadata support
            self.index.index(tokenized_corpus)
        
        if self.config.index_path:
            self.save_index()
    
    def load_index(self) -> None:
        """Load BM25s index from file."""
        if not self.config.index_path:
            raise ValueError("Index path not configured")
        
        index_path = Path(self.config.index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self._documents = data['documents']
    
    def save_index(self) -> None:
        """Save BM25s index to file."""
        if not self.config.index_path:
            raise ValueError("Index path not configured")
        
        if not self.index:
            raise ValueError("No index to save")
        
        index_path = Path(self.config.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'index': self.index,
            'documents': self._documents
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(data, f)
    
    def get_documents(self) -> List[BM25sDocument]:
        """Get indexed documents."""
        return self._documents.copy()


class BM25sSearchService:
    """Service for BM25s search operations."""
    
    def __init__(self, index_service: BM25sIndexService):
        self.index_service = index_service
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[BM25sResult]:
        """Search documents using BM25s algorithm with optional metadata filtering."""
        if not self.index_service.index:
            raise ValueError("Index not available. Create or load index first.")
        
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        tokenized_query = bm25s.tokenize(query)
        
        # Use metadata filtering if available (bm25s-j 0.2.0+ feature)
        try:
            if metadata_filter:
                result = self.index_service.index.retrieve(
                    tokenized_query, 
                    k=top_k,
                    filter=metadata_filter
                )
            else:
                result = self.index_service.index.retrieve(
                    tokenized_query, 
                    k=top_k
                )
            
            # Handle different return formats (2 or 3 values)
            if len(result) == 2:
                scores, indices = result
            elif len(result) == 3:
                scores, indices, metadata = result
            else:
                scores, indices = result[0], result[1]
                
        except TypeError:
            # Fallback for older versions - retrieve all and filter afterwards
            result = self.index_service.index.retrieve(
                tokenized_query, 
                k=top_k * 3 if metadata_filter else top_k
            )
            if len(result) == 2:
                scores, indices = result
            elif len(result) == 3:
                scores, indices, metadata = result
            else:
                scores, indices = result[0], result[1]
        
        results = []
        documents = self.index_service.get_documents()
        
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Convert numpy types to Python types
            idx = int(idx)
            if idx < len(documents):
                document = documents[idx]
                
                # Apply post-filtering if bm25s doesn't support native filtering
                if metadata_filter and not self._supports_native_filtering():
                    if not self._matches_filter(document.metadata, metadata_filter):
                        continue
                
                result = BM25sResult(
                    document=document,
                    score=float(score),
                    rank=rank + 1
                )
                results.append(result)
                
                # Stop if we have enough results after filtering
                if len(results) >= top_k:
                    break
        
        return sorted(results)
    
    def _supports_native_filtering(self) -> bool:
        """Check if bm25s supports native metadata filtering."""
        try:
            # Try to access retrieve method signature
            import inspect
            sig = inspect.signature(self.index_service.index.retrieve)
            return 'filter' in sig.parameters
        except:
            return False
    
    def _matches_filter(self, metadata: Optional[Dict[str, Any]], filter_dict: Dict[str, Any]) -> bool:
        """Check if document metadata matches the filter criteria."""
        if not metadata:
            return not filter_dict  # Empty metadata matches empty filter
        
        for key, value in filter_dict.items():
            doc_value = metadata.get(key)
            
            if isinstance(value, list):
                # List means "one of these values"
                if doc_value not in value:
                    return False
            elif isinstance(value, dict):
                # Handle operators like {"$gte": 0.8}
                if not self._apply_operator_filter(doc_value, value):
                    return False
            else:
                # Exact match
                if doc_value != value:
                    return False
        
        return True
    
    def _apply_operator_filter(self, doc_value: Any, operator_dict: Dict[str, Any]) -> bool:
        """Apply operator-based filtering."""
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
            else:
                # Unsupported operator - skip
                continue
        
        return True
    
    def batch_search(
        self, 
        queries: List[str], 
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[List[BM25sResult]]:
        """Perform batch search for multiple queries with optional metadata filtering."""
        if not queries:
            raise ValueError("Queries list cannot be empty")
        
        results = []
        for query in queries:
            query_results = self.search(query, top_k, metadata_filter)
            results.append(query_results)
        
        return results
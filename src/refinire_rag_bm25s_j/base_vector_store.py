"""Base VectorStore implementation independent of langchain."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BaseDocument:
    """Simple document representation."""
    page_content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore."""
        pass
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[BaseDocument],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vectorstore."""
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[BaseDocument]:
        """Return docs most similar to query."""
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[BaseDocument, float]]:
        """Return docs and scores most similar to query."""
        pass
    
    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by IDs."""
        pass
    
    @classmethod
    @abstractmethod
    def from_texts(
        cls,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "BaseVectorStore":
        """Create VectorStore from list of texts."""
        pass
    
    @classmethod
    @abstractmethod
    def from_documents(
        cls,
        documents: List[BaseDocument],
        **kwargs: Any,
    ) -> "BaseVectorStore":
        """Create VectorStore from list of documents."""
        pass
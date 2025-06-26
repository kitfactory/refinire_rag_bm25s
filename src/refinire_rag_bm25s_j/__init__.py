"""BM25s VectorStore and KeywordSearch plugin for refinire-rag."""

__version__ = "0.0.1"

# Always export standalone version
from .bm25s_store import BM25sStore
from .base_vector_store import BaseDocument

# Short alias for convenience
# BM25sStore is now the main class name

__all__ = ["BM25sStore", "BaseDocument"]

# Try to export langchain-compatible version if available
try:
    from .vector_store import BM25sVectorStore
    __all__.append("BM25sVectorStore")
except ImportError:
    # langchain-core not available, use standalone version as default
    BM25sVectorStore = BM25sStore

# Try to export KeywordSearch version if available
try:
    from .keyword_store import BM25sKeywordStore
    __all__.append("BM25sKeywordStore")
except ImportError:
    # refinire-rag not available, skip KeywordSearch export
    pass
# refinire-rag-bm25s-j

BM25s VectorStore plugin for refinire-rag - A fast, efficient text search solution using BM25s ranking algorithm.

## Overview

This plugin provides BM25s database functionality as a VectorStore subclass for refinire-rag. While BM25s is not technically a vector search, it provides equivalent usage patterns and is implemented as a VectorStore subclass for seamless integration.

**Key Dependencies:**
- `refinire-rag` - RAG framework integration
- `bm25s-j` - Fast BM25s implementation

## Features

-  **Fast keyword-based retrieval** - No embedding computation needed
-  **Metadata filtering** - Advanced filtering with comparison operators (BM25s-j 0.2.0+)
-  **Index persistence** - Save/load indexes for production use
-  **Memory efficient** - Optimized for large document collections
-  **Deterministic results** - Explainable and reproducible search
-  **Production ready** - Comprehensive error handling and logging

## Installation

```bash
# Basic installation
uv add refinire-rag-bm25s-j

# With refinire-rag integration
uv add refinire-rag-bm25s-j[rag]

# For development
uv add refinire-rag-bm25s-j[dev]
```

## Quick Start

```python
from refinire_rag_bm25s_j import BM25sStore
from refinire_rag_bm25s_j.models import BM25sConfig

# Create configuration
config = BM25sConfig(
    k1=1.2,          # Term frequency saturation
    b=0.75,          # Length normalization  
    epsilon=0.25,    # IDF cutoff
    index_path="./data/bm25s_index.pkl"
)

# Initialize store
store = BM25sStore(config=config)

# Add documents
documents = [
    "Python is a powerful programming language for data science.",
    "Machine learning algorithms can be implemented efficiently.",
    "BM25 is a ranking function used in information retrieval."
]

store.add_texts(documents)

# Search
results = store.similarity_search("Python programming", k=2)
for doc in results:
    print(f"Score: {doc.metadata['score']:.3f}")
    print(f"Content: {doc.page_content}")
```

## Configuration

### BM25s Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `k1` | Term frequency saturation | 1.2 | 1.2-1.5 |
| `b` | Length normalization | 0.75 | 0.7-0.8 |
| `epsilon` | IDF cutoff parameter | 0.25 | 0.1-0.3 |
| `index_path` | Index save/load path | None | Set for persistence |

### Use Case Tuning

```python
# Technical documentation
config = BM25sConfig(k1=1.5, b=0.8, epsilon=0.1)

# General knowledge base  
config = BM25sConfig(k1=1.2, b=0.75, epsilon=0.25)

# Legal/medical texts
config = BM25sConfig(k1=1.0, b=0.9, epsilon=0.1)
```

## Advanced Features

### Metadata Filtering (BM25s-j 0.2.0+)

```python
# Add documents with metadata
documents = ["Document content..."]
metadata = [{"category": "tech", "year": 2024, "rating": 4.5}]
store.add_texts(documents, metadatas=metadata)

# Basic filtering
results = store.similarity_search(
    "search query",
    filter={"category": "tech"}
)

# Advanced filtering with operators
results = store.similarity_search(
    "search query", 
    filter={
        "year": {"$gte": 2023},
        "rating": {"$gt": 4.0},
        "category": {"$in": ["tech", "science"]}
    }
)
```

### Supported Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$gt` | Greater than | `{"score": {"$gt": 0.8}}` |
| `$gte` | Greater than or equal | `{"year": {"$gte": 2023}}` |
| `$lt` | Less than | `{"priority": {"$lt": 5}}` |
| `$lte` | Less than or equal | `{"rating": {"$lte": 3.0}}` |
| `$in` | Value in list | `{"type": {"$in": ["doc", "guide"]}}` |
| `$nin` | Value not in list | `{"status": {"$nin": ["draft"]}}` |
| `$ne` | Not equal | `{"private": {"$ne": true}}` |
| `$exists` | Field exists | `{"author": {"$exists": true}}` |

### Index Persistence

```python
# Save index automatically
config = BM25sConfig(index_path="./data/my_index.pkl")
store = BM25sStore(config=config)
store.add_texts(documents)  # Index auto-saved

# Manual save/load
store.index_service.save_index()
store.index_service.load_index()
```

## Examples

Comprehensive examples are available in `src/examples/`:

- **`basic_usage.py`** - Simple setup and search
- **`integration_example.py`** - LangChain Document integration  
- **`rag_pipeline_example.py`** - Complete RAG system
- **`metadata_filtering_example.py`** - Advanced filtering
- **`production_rag_example.py`** - Production deployment
- **`hybrid_search_example.py`** - Combine with semantic search

```bash
# Run examples
python src/examples/basic_usage.py
python src/examples/rag_pipeline_example.py
```

## API Reference

### BM25sStore

Main class providing VectorStore interface:

```python
class BM25sStore(BaseVectorStore):
    def add_texts(texts, metadatas=None, ids=None) -> List[str]
    def add_documents(documents, ids=None) -> List[str]  
    def similarity_search(query, k=4, filter=None) -> List[BaseDocument]
    def similarity_search_with_score(query, k=4, filter=None) -> List[Tuple[BaseDocument, float]]
    def delete(ids) -> bool
    
    @classmethod
    def from_texts(texts, metadatas=None, config=None) -> BM25sStore
    
    @classmethod  
    def from_documents(documents, config=None) -> BM25sStore
```

### BM25sConfig

Configuration model with validation:

```python
class BM25sConfig(BaseModel):
    k1: float = 1.2           # Term frequency saturation
    b: float = 0.75           # Length normalization  
    epsilon: float = 0.25     # IDF cutoff
    index_path: Optional[str] = None  # Save/load path
```

## Performance Guidelines

### When to Use BM25s

**Ideal for:**
- Technical documentation and API references
- FAQ systems with exact keyword matching  
- Legal and medical document search
- Code snippet retrieval
- Any scenario requiring explainable results

**Consider hybrid approach for:**
- General knowledge questions
- Conceptual/semantic queries  
- Cross-lingual search
- Synonym and paraphrase handling

### Optimization Tips

1. **Document chunking**: 500-1500 tokens per chunk with 10-20% overlap
2. **Index management**: Save indexes after creation, implement incremental updates
3. **Query preprocessing**: Normalize and expand queries for better results
4. **Metadata strategy**: Use filtering to reduce search space before ranking

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/e2e/
```

## Development

```bash
# Install development dependencies
uv sync --group dev

# Run linting and formatting
uv run ruff check src/
uv run ruff format src/

# Type checking
uv run mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## License

[License information]

## Support

- Documentation: See `docs/` directory
- Examples: See `src/examples/` directory  
- Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

**refinire-rag-bm25s-j** - Fast, efficient, and production-ready BM25s search for RAG applications.
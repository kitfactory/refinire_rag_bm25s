# BM25s Plugin Examples

This directory contains comprehensive examples demonstrating how to use the BM25s plugin with refinire-rag, supporting both VectorStore and KeywordSearch interfaces.

## Examples Overview

### KeywordSearch Plugin Examples

### K1. `keyword_search_example.py`
**Complete KeywordSearch Functionality Demo**
- Document indexing and keyword-based search
- Metadata filtering with BM25s
- Document management operations (add, update, delete)
- Search performance analysis

```bash
python examples/keyword_search_example.py
```

### K2. `refinire_rag_keyword_integration.py`
**Integration with refinire-rag Framework**
- Using BM25s as DocumentProcessor
- Multi-language content processing
- Advanced metadata filtering patterns
- Batch processing workflows

```bash
python examples/refinire_rag_keyword_integration.py
```

### K3. `plugin_registration_example.py`
**Plugin Registration and Discovery**
- Plugin discovery mechanism demonstration
- Environment-based configuration
- Performance tuning examples
- Multi-configuration comparisons

```bash
python examples/plugin_registration_example.py
```

### VectorStore Examples

### 1. `basic_usage.py`
**Basic BM25s VectorStore Usage**
- Simple document indexing and search
- Basic configuration and setup
- Document management (add, search, delete)
- Score interpretation

```bash
python examples/basic_usage.py
```

### 2. `integration_example.py`
**Integration with LangChain Documents**
- Working with LangChain Document objects
- Metadata handling and filtering
- Index persistence across sessions
- Batch operations

```bash
python examples/integration_example.py
```

### 3. `rag_pipeline_example.py`
**RAG Pipeline with BM25s**
- Complete RAG system setup
- Query processing and context building
- Multiple query types demonstration
- Performance characteristics analysis

```bash
python examples/rag_pipeline_example.py
```

### 4. `hybrid_search_example.py`
**Hybrid Search Implementation**
- Combining BM25s with semantic search
- Score normalization and weighting
- Query routing strategies
- Performance comparisons

```bash
python examples/hybrid_search_example.py
```

### 5. `production_rag_example.py`
**Production-Ready RAG System**
- Scalable document ingestion
- Performance monitoring and metrics
- Error handling and logging
- Production deployment guidelines

```bash
python examples/production_rag_example.py
```

### 6. `metadata_filtering_example.py`
**Metadata Filtering with BM25s-j 0.2.0+**
- Basic and advanced metadata filtering
- Comparison operators ($gte, $lt, etc.)
- Real-world filtering scenarios
- Performance impact analysis

```bash
python examples/metadata_filtering_example.py
```

## Prerequisites

Install the required dependencies:

```bash
# Basic functionality
uv add refinire-rag bm25s-j

# For production examples
uv add fastapi uvicorn  # if using web APIs
```

## Quick Start

1. **Start with basic usage** to understand core concepts:
   ```bash
   python examples/basic_usage.py
   ```

2. **Explore RAG pipeline** for practical applications:
   ```bash
   python examples/rag_pipeline_example.py
   ```

3. **Test production features** for real deployments:
   ```bash
   python examples/production_rag_example.py
   ```

## Configuration Guidelines

### BM25s Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `k1` | Term frequency saturation | 1.2 (general), 1.5 (technical docs) |
| `b` | Length normalization | 0.75 (general), 0.8 (varied lengths) |
| `epsilon` | IDF cutoff | 0.25 (general), 0.1 (strict filtering) |

### Use Case Recommendations

| Use Case | Configuration | Example |
|----------|---------------|---------|
| **Technical Documentation** | k1=1.5, b=0.8, ε=0.1 | API docs, code search |
| **General Knowledge** | k1=1.2, b=0.75, ε=0.25 | FAQs, general content |
| **Legal/Medical** | k1=1.0, b=0.9, ε=0.1 | Precise terminology |
| **News/Articles** | k1=1.3, b=0.7, ε=0.3 | Varied content lengths |

## Performance Characteristics

### BM25s Strengths
- ✅ Fast keyword-based retrieval
- ✅ No embedding computation needed
- ✅ Excellent for exact term matching
- ✅ Deterministic and explainable results
- ✅ Memory efficient
- ✅ Works well with technical terminology

### BM25s Limitations
- ❌ Limited semantic understanding
- ❌ Vocabulary mismatch issues
- ❌ Poor with synonyms and paraphrases
- ❌ No cross-lingual capabilities

### When to Use BM25s

**Ideal for:**
- Technical documentation search
- Code snippet and API reference retrieval
- FAQ systems with exact keyword matching
- Legal document search
- Medical literature with precise terminology
- Any scenario requiring explainable search results

**Consider hybrid approach for:**
- General knowledge questions
- Conceptual queries
- Cross-lingual search
- Synonym and paraphrase handling

## Integration Patterns

### 1. Pure BM25s RAG
```python
from refinire_rag_bm25s_j import BM25sStore
from refinire_rag_bm25s_j.models import BM25sConfig

config = BM25sConfig(k1=1.2, b=0.75)
store = BM25sStore(config=config)
# Use for keyword-heavy domains
```

### 2. Hybrid Search
```python
# Combine BM25s with semantic search
bm25s_results = bm25s_store.similarity_search(query, k=10)
semantic_results = semantic_store.similarity_search(query, k=10)
# Merge and rerank results
```

### 3. Query Routing
```python
def route_query(query):
    if has_technical_terms(query):
        return bm25s_search(query)
    else:
        return semantic_search(query)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain_core'
   ```
   **Solution:** Install refinire-rag which includes langchain-core

2. **Index Not Found**
   ```
   FileNotFoundError: Index file not found
   ```
   **Solution:** Check index_path configuration or create new index

3. **Poor Search Results**
   **Solution:** Adjust BM25s parameters or consider hybrid search

### Performance Optimization

1. **Chunking Strategy**
   - Use 500-1500 tokens per chunk
   - Overlap 10-20% between chunks
   - Consider document structure

2. **Index Management**
   - Save index after initial creation
   - Implement incremental updates
   - Monitor index size and performance

3. **Query Optimization**
   - Preprocess queries (normalize, expand)
   - Use metadata filtering (BM25s-j 0.2.0+)
   - Implement result caching

## New in BM25s-j 0.2.0+

### Metadata Filtering Features
- ✅ **Basic filtering**: `{"category": "tech"}`
- ✅ **Multiple criteria**: `{"category": "tech", "year": 2024}`
- ✅ **List filtering**: `{"language": ["python", "java"]}`
- ✅ **Comparison operators**: `{"rating": {"$gte": 4.0}}`
- ✅ **Existence checks**: `{"author": {"$exists": True}}`
- ✅ **Exclusion**: `{"status": {"$ne": "deprecated"}}`

### Filter Operators
| Operator | Description | Example |
|----------|-------------|---------|
| `$gt` | Greater than | `{"score": {"$gt": 0.8}}` |
| `$gte` | Greater than or equal | `{"year": {"$gte": 2023}}` |
| `$lt` | Less than | `{"difficulty": {"$lt": 5}}` |
| `$lte` | Less than or equal | `{"priority": {"$lte": 3}}` |
| `$in` | Value in list | `{"category": {"$in": ["tech", "science"]}}` |
| `$nin` | Value not in list | `{"status": {"$nin": ["draft", "deleted"]}}` |
| `$ne` | Not equal | `{"type": {"$ne": "private"}}` |
| `$exists` | Field exists/doesn't exist | `{"metadata": {"$exists": True}}` |

## Contributing

To add new examples:

1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Add error handling and logging
4. Update this README with the new example
5. Test with different query types

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the main project documentation
- Submit issues to the project repository
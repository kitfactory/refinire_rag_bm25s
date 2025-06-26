"""RAG pipeline example using BM25s VectorStore with refinire-rag."""

import os
from typing import List

try:
    from refinire import Refinire
    from refinire.config import Config
    from refinire.rag import RAGPipeline
    from refinire.document_loader import DocumentLoader
    from refinire.text_splitter import TextSplitter
except ImportError:
    print("This example requires refinire-rag. Install it with: uv add refinire-rag")
    exit(1)

from refinire_rag_bm25s_j import BM25sStore
from refinire_rag_bm25s_j.models import BM25sConfig


def create_sample_documents() -> List[str]:
    """Create sample documents for the RAG system."""
    documents = [
        # AI and Machine Learning
        """
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
        that can perform tasks that typically require human intelligence. Machine Learning is a subset of AI 
        that focuses on algorithms that can learn and improve from experience without being explicitly programmed.
        """,
        
        """
        Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers 
        to model and understand complex patterns in data. It has been particularly successful in image recognition, 
        natural language processing, and speech recognition tasks.
        """,
        
        """
        Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers 
        and human language. It involves developing algorithms and models that can understand, interpret, 
        and generate human language in a valuable way.
        """,
        
        # Python Programming
        """
        Python is a high-level, interpreted programming language known for its simplicity and readability. 
        It supports multiple programming paradigms including procedural, object-oriented, and functional programming. 
        Python is widely used in web development, data science, artificial intelligence, and automation.
        """,
        
        """
        FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python 
        type hints. It provides automatic API documentation, high performance, and is easy to use. FastAPI 
        is built on top of Starlette for the web parts and Pydantic for the data parts.
        """,
        
        """
        Pandas is a powerful data manipulation and analysis library for Python. It provides data structures 
        like DataFrame and Series that make it easy to work with structured data. Pandas is essential for 
        data cleaning, transformation, and analysis in data science projects.
        """,
        
        # Data Science
        """
        Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, 
        and systems to extract knowledge and insights from structured and unstructured data. It combines 
        statistics, machine learning, and domain expertise to solve complex problems.
        """,
        
        """
        Feature Engineering is the process of selecting, modifying, or creating new features from raw data 
        to improve the performance of machine learning models. It involves domain knowledge and creativity 
        to transform data into a format that better represents the underlying problem.
        """,
        
        """
        Cross-validation is a statistical method used to estimate the performance of machine learning models. 
        It involves dividing the dataset into multiple subsets, training the model on some subsets, 
        and testing on others to get a more robust estimate of model performance.
        """,
        
        # Vector Databases and Search
        """
        Vector databases are specialized databases designed to store and query high-dimensional vectors 
        efficiently. They are essential for applications like semantic search, recommendation systems, 
        and similarity matching in machine learning and AI applications.
        """,
        
        """
        BM25 (Best Matching 25) is a ranking function used in information retrieval to estimate the relevance 
        of documents to a given search query. It is based on the probabilistic retrieval framework and 
        is widely used in search engines and document retrieval systems.
        """,
        
        """
        Semantic search goes beyond keyword matching to understand the intent and contextual meaning of 
        search queries. It uses techniques like word embeddings and transformer models to find semantically 
        similar content even when exact keywords don't match.
        """
    ]
    
    return documents


def setup_rag_with_bm25s():
    """Set up RAG pipeline using BM25s VectorStore."""
    
    print("=== Setting up RAG Pipeline with BM25s VectorStore ===\n")
    
    # Configure BM25s for optimal performance
    bm25s_config = BM25sConfig(
        k1=1.5,  # Higher term frequency scaling for technical documents
        b=0.8,   # Strong length normalization
        epsilon=0.1,  # Strict IDF filtering
        index_path="./data/rag_bm25s_index.pkl"
    )
    
    # Create BM25s VectorStore
    vector_store = BM25sStore(config=bm25s_config)
    
    # Get sample documents
    documents = create_sample_documents()
    
    # Add documents to vector store with metadata
    print("Adding documents to BM25s VectorStore...")
    metadatas = [
        {"category": "ai", "topic": "artificial_intelligence"},
        {"category": "ai", "topic": "deep_learning"},
        {"category": "ai", "topic": "nlp"},
        {"category": "programming", "topic": "python_basics"},
        {"category": "programming", "topic": "fastapi"},
        {"category": "programming", "topic": "pandas"},
        {"category": "data_science", "topic": "overview"},
        {"category": "data_science", "topic": "feature_engineering"},
        {"category": "data_science", "topic": "cross_validation"},
        {"category": "search", "topic": "vector_databases"},
        {"category": "search", "topic": "bm25"},
        {"category": "search", "topic": "semantic_search"}
    ]
    
    doc_ids = vector_store.add_texts(documents, metadatas)
    print(f"Successfully added {len(doc_ids)} documents to the vector store.\n")
    
    return vector_store


def demonstrate_rag_queries(vector_store: BM25sStore):
    """Demonstrate various RAG queries using BM25s."""
    
    print("=== RAG Query Examples ===\n")
    
    # Define test queries with different complexity levels
    queries = [
        {
            "query": "What is machine learning?",
            "description": "Basic AI concept query"
        },
        {
            "query": "How to build APIs with Python?",
            "description": "Programming-specific query"
        },
        {
            "query": "Feature engineering techniques for data science",
            "description": "Technical data science query"
        },
        {
            "query": "BM25 algorithm for document retrieval",
            "description": "Information retrieval query"
        },
        {
            "query": "Difference between semantic search and keyword search",
            "description": "Comparative concept query"
        }
    ]
    
    for i, query_info in enumerate(queries, 1):
        query = query_info["query"]
        description = query_info["description"]
        
        print(f"Query {i}: {description}")
        print(f"Question: '{query}'")
        print("-" * 60)
        
        # Retrieve relevant documents
        relevant_docs = vector_store.similarity_search_with_score(query, k=3)
        
        if relevant_docs:
            # Simulate RAG response generation
            context_pieces = []
            
            for rank, (doc, score) in enumerate(relevant_docs, 1):
                print(f"Source {rank} (Relevance: {score:.3f}):")
                print(f"  Category: {doc.metadata.get('category', 'unknown')}")
                print(f"  Topic: {doc.metadata.get('topic', 'unknown')}")
                print(f"  Content: {doc.page_content[:150]}...")
                print()
                
                # Extract content for context
                context_pieces.append(doc.page_content.strip())
            
            # Combine context for RAG
            combined_context = "\n\n".join(context_pieces)
            
            print("Generated RAG Context:")
            print(f"  Total context length: {len(combined_context)} characters")
            print(f"  Number of sources: {len(context_pieces)}")
            print()
            
            # In a real RAG system, this context would be sent to an LLM
            # along with the query to generate a comprehensive response
            print("💡 In a complete RAG system, this context would be sent to an LLM")
            print("   to generate a comprehensive answer based on the retrieved knowledge.\n")
        
        else:
            print("No relevant documents found.\n")
        
        print("=" * 80 + "\n")


def demonstrate_advanced_features(vector_store: BM25sStore):
    """Demonstrate advanced BM25s features."""
    
    print("=== Advanced BM25s Features ===\n")
    
    # 1. Category-based filtering (simulated through metadata analysis)
    print("1. Category-based Analysis:")
    query = "Python programming frameworks"
    results = vector_store.similarity_search_with_score(query, k=5)
    
    categories = {}
    for doc, score in results:
        category = doc.metadata.get('category', 'unknown')
        if category not in categories:
            categories[category] = []
        categories[category].append((doc, score))
    
    for category, docs in categories.items():
        print(f"   {category.title()}: {len(docs)} relevant documents")
        for doc, score in docs[:2]:  # Show top 2 per category
            print(f"     - {doc.metadata.get('topic', 'unknown')} (score: {score:.3f})")
    print()
    
    # 2. Multi-query expansion
    print("2. Multi-Query Analysis:")
    related_queries = [
        "machine learning algorithms",
        "artificial intelligence applications",
        "neural network architectures"
    ]
    
    all_results = {}
    for q in related_queries:
        results = vector_store.similarity_search_with_score(q, k=2)
        all_results[q] = results
    
    for query, results in all_results.items():
        print(f"   Query: '{query}'")
        for doc, score in results:
            topic = doc.metadata.get('topic', 'unknown')
            print(f"     - {topic} (score: {score:.3f})")
        print()
    
    # 3. Document similarity analysis
    print("3. Document Clustering by Topic:")
    all_docs = vector_store.similarity_search("", k=12)  # Get all documents
    topics = {}
    
    for doc in all_docs:
        category = doc.metadata.get('category', 'unknown')
        if category not in topics:
            topics[category] = []
        topics[category].append(doc.metadata.get('topic', 'unknown'))
    
    for category, topic_list in topics.items():
        print(f"   {category.title()} ({len(topic_list)} documents):")
        for topic in topic_list:
            print(f"     - {topic}")
        print()


def performance_comparison():
    """Compare BM25s performance characteristics."""
    
    print("=== BM25s Performance Characteristics ===\n")
    
    print("BM25s Advantages for RAG:")
    print("✓ Fast keyword-based retrieval")
    print("✓ No embedding computation required")
    print("✓ Excellent for exact term matching")
    print("✓ Lightweight and memory efficient")
    print("✓ Deterministic and explainable results")
    print("✓ Works well with domain-specific terminology")
    print()
    
    print("Best Use Cases:")
    print("• Technical documentation search")
    print("• Code snippet retrieval")
    print("• FAQ and knowledge base queries")
    print("• Legal and medical document search")
    print("• Exact keyword matching scenarios")
    print()
    
    print("Hybrid Approach Recommendations:")
    print("• Combine BM25s with semantic search for best results")
    print("• Use BM25s for high-precision keyword queries")
    print("• Use semantic search for conceptual and contextual queries")
    print("• Implement query routing based on query type")
    print()


def main():
    """Main function demonstrating RAG with BM25s."""
    
    print("🔍 BM25s VectorStore RAG Pipeline Demo")
    print("=" * 50 + "\n")
    
    # Set up the RAG system
    vector_store = setup_rag_with_bm25s()
    
    # Demonstrate various query scenarios
    demonstrate_rag_queries(vector_store)
    
    # Show advanced features
    demonstrate_advanced_features(vector_store)
    
    # Performance insights
    performance_comparison()
    
    print("Demo completed! 🎉")
    print("\nNext steps:")
    print("1. Integrate with your preferred LLM (OpenAI, Claude, etc.)")
    print("2. Add document chunking and preprocessing")
    print("3. Implement hybrid search with semantic vectors")
    print("4. Add query expansion and reranking")


if __name__ == "__main__":
    main()
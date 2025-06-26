"""Integration example with refinire-rag pipeline."""

from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    print("This example requires langchain-core. Install refinire-rag to get it.")
    exit(1)

from refinire_rag_bm25s_j import BM25sStore
from refinire_rag_bm25s_j.models import BM25sConfig


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    documents = [
        Document(
            page_content="Artificial intelligence is transforming various industries including healthcare, finance, and transportation.",
            metadata={"source": "ai_article.txt", "category": "technology", "date": "2024-01-15"}
        ),
        Document(
            page_content="Machine learning models require large datasets and significant computational resources for training.",
            metadata={"source": "ml_guide.pdf", "category": "technology", "date": "2024-02-01"}
        ),
        Document(
            page_content="Natural language processing enables computers to understand and generate human language.",
            metadata={"source": "nlp_intro.md", "category": "ai", "date": "2024-01-20"}
        ),
        Document(
            page_content="Deep learning neural networks have revolutionized computer vision and speech recognition.",
            metadata={"source": "deep_learning.txt", "category": "ai", "date": "2024-02-10"}
        ),
        Document(
            page_content="Data preprocessing is a crucial step in any machine learning pipeline.",
            metadata={"source": "data_prep.pdf", "category": "data_science", "date": "2024-01-25"}
        )
    ]
    return documents


def demonstrate_vector_store_integration():
    """Demonstrate BM25s VectorStore integration with langchain documents."""
    
    print("=== BM25s VectorStore Integration Example ===\n")
    
    # Configuration for BM25s
    config = BM25sConfig(
        k1=1.5,  # Higher k1 for more aggressive term frequency scaling
        b=0.8,   # Higher b for more length normalization
        epsilon=0.1,  # Lower epsilon for stricter IDF filtering
        index_path="./data/integration_index.pkl"
    )
    
    # Create vector store using class method
    documents = create_sample_documents()
    vector_store = BM25sStore.from_documents(documents, config=config)
    
    print(f"Created vector store with {len(documents)} documents\n")
    
    # Test different types of queries
    queries = [
        "machine learning datasets",
        "natural language processing",
        "artificial intelligence healthcare",
        "neural networks computer vision"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 50)
        
        # Get top 2 results with scores
        results = vector_store.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"Result {i} (Score: {score:.3f}):")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Source: {doc.metadata.get('source', 'unknown')}")
            print(f"  Category: {doc.metadata.get('category', 'unknown')}")
            print()
        
        print()
    
    # Demonstrate batch addition
    print("=== Adding More Documents ===")
    new_texts = [
        "Computer vision algorithms can detect objects in images and videos.",
        "Reinforcement learning teaches agents to make decisions through trial and error."
    ]
    new_metadatas = [
        {"source": "cv_tutorial.txt", "category": "computer_vision"},
        {"source": "rl_basics.pdf", "category": "reinforcement_learning"}
    ]
    
    new_ids = vector_store.add_texts(new_texts, new_metadatas)
    print(f"Added {len(new_ids)} new documents: {new_ids}\n")
    
    # Search again to see updated results
    test_query = "computer vision algorithms"
    print(f"Updated search for: '{test_query}'")
    updated_results = vector_store.similarity_search(test_query, k=3)
    
    for i, doc in enumerate(updated_results, 1):
        print(f"{i}. {doc.page_content[:80]}...")
        print(f"   Score: {doc.metadata.get('score', 'N/A')}")
        print()
    
    # Demonstrate MaxMarginalRelevance search (falls back to similarity search)
    print("=== Max Marginal Relevance Search ===")
    mmr_results = vector_store.max_marginal_relevance_search(
        "machine learning artificial intelligence", 
        k=3,
        fetch_k=5
    )
    
    for i, doc in enumerate(mmr_results, 1):
        print(f"{i}. {doc.page_content[:80]}...")
        print(f"   Category: {doc.metadata.get('category', 'unknown')}")
        print()


def demonstrate_persistence():
    """Demonstrate index persistence across sessions."""
    
    print("=== Index Persistence Demo ===\n")
    
    config = BM25sConfig(index_path="./data/persistent_index.pkl")
    
    # First session: Create and save
    print("Session 1: Creating and saving index...")
    vector_store1 = BM25sStore(config=config)
    
    texts = ["First document", "Second document", "Third document"]
    vector_store1.add_texts(texts)
    print("Index created and saved.\n")
    
    # Second session: Load existing index
    print("Session 2: Loading existing index...")
    vector_store2 = BM25sStore(config=config)
    
    # Search to verify loaded index
    results = vector_store2.similarity_search("document", k=3)
    print(f"Loaded index successfully. Found {len(results)} documents.")
    
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
    
    print()


if __name__ == "__main__":
    demonstrate_vector_store_integration()
    demonstrate_persistence()
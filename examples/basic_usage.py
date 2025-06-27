"""Basic usage example for BM25s VectorStore."""

from refinire_rag_bm25s_j import BM25sStore


def main():
    """Demonstrate basic usage of BM25s VectorStore."""
    
    # Initialize vector store with unified configuration approach
    # Configuration priority: kwargs > environment variables > defaults
    vector_store = BM25sStore(
        k1=1.2,
        b=0.75,
        epsilon=0.25,
        index_path="./data/bm25s_index.pkl"
    )
    
    # Sample documents
    documents = [
        "Python is a powerful programming language used for data science.",
        "Machine learning algorithms can be implemented in Python.",
        "Natural language processing is a subset of artificial intelligence.",
        "Vector databases are essential for semantic search applications.",
        "BM25 is a ranking function used in information retrieval systems."
    ]
    
    # Add documents to the vector store
    print("Adding documents to BM25s VectorStore...")
    doc_ids = vector_store.add_texts(
        texts=documents,
        metadatas=[
            {"category": "programming", "language": "python"},
            {"category": "ml", "language": "python"},
            {"category": "nlp", "domain": "ai"},
            {"category": "databases", "type": "vector"},
            {"category": "search", "algorithm": "bm25"}
        ]
    )
    print(f"Added {len(doc_ids)} documents with IDs: {doc_ids}")
    
    # Perform similarity search
    print("\nPerforming similarity search...")
    query = "Python machine learning"
    results = vector_store.similarity_search(query, k=3)
    
    print(f"Query: '{query}'")
    print("Top 3 results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. Score: {doc.metadata.get('score', 'N/A'):.3f}")
        print(f"   Content: {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")
        print()
    
    # Perform similarity search with scores
    print("Performing similarity search with explicit scores...")
    results_with_scores = vector_store.similarity_search_with_score(query, k=2)
    
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content}")
        print()
    
    # Demonstrate document deletion
    print("Deleting a document...")
    deleted = vector_store.delete([doc_ids[0]])
    print(f"Document deletion successful: {deleted}")
    
    # Search again to verify deletion
    print("\nSearching after deletion...")
    results_after_deletion = vector_store.similarity_search(query, k=5)
    print(f"Found {len(results_after_deletion)} documents after deletion")


if __name__ == "__main__":
    main()
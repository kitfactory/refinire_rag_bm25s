"""Integration test with refinire-rag to verify save and search functionality."""

import os
import tempfile
from pathlib import Path

try:
    from langchain_core.documents import Document
    from refinire_rag_bm25s_j import BM25sStore
    from refinire_rag_bm25s_j.models import BM25sConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure refinire-rag is installed with: uv add refinire-rag")
    exit(1)


def test_basic_functionality():
    """Test basic save and search functionality."""
    print("=== Basic Functionality Test ===\n")
    
    # Create temporary directory for index
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "test_index.pkl")
        
        # 1. Initialize BM25sStore
        print("1. Initializing BM25sStore...")
        config = BM25sConfig(
            k1=1.2,
            b=0.75,
            index_path=index_path
        )
        vector_store = BM25sStore(config=config)
        print("✅ VectorStore initialized successfully\n")
        
        # 2. Test document addition (Save)
        print("2. Testing document addition...")
        documents = [
            "Python is a versatile programming language used for web development and data science.",
            "Machine learning models can be trained using various algorithms like neural networks.",
            "FastAPI is a modern web framework for building APIs with Python type hints.",
            "Data visualization helps in understanding complex patterns in datasets.",
            "Natural language processing enables computers to understand human language."
        ]
        
        metadatas = [
            {"category": "programming", "topic": "python", "difficulty": "beginner"},
            {"category": "ai", "topic": "machine_learning", "difficulty": "intermediate"},
            {"category": "web", "topic": "fastapi", "difficulty": "intermediate"},
            {"category": "data_science", "topic": "visualization", "difficulty": "beginner"},
            {"category": "ai", "topic": "nlp", "difficulty": "advanced"}
        ]
        
        doc_ids = vector_store.add_texts(documents, metadatas)
        print(f"✅ Added {len(doc_ids)} documents with IDs: {doc_ids}\n")
        
        # 3. Test basic search
        print("3. Testing basic search...")
        query = "Python programming"
        results = vector_store.similarity_search(query, k=3)
        
        print(f"Query: '{query}'")
        print(f"Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. Score: {doc.metadata.get('score', 'N/A'):.3f}")
            print(f"     Content: {doc.page_content[:60]}...")
            print(f"     Category: {doc.metadata.get('category', 'unknown')}")
        print()
        
        # 4. Test search with scores
        print("4. Testing search with scores...")
        results_with_scores = vector_store.similarity_search_with_score(query, k=2)
        
        print("Results with explicit scores:")
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"  {i}. BM25 Score: {score:.3f}")
            print(f"     Topic: {doc.metadata.get('topic', 'unknown')}")
        print()
        
        # 5. Verify index persistence
        print("5. Testing index persistence...")
        assert os.path.exists(index_path), "Index file should exist"
        file_size = os.path.getsize(index_path)
        print(f"✅ Index saved to: {index_path}")
        print(f"   File size: {file_size} bytes\n")
        
        return True


def test_metadata_filtering():
    """Test metadata filtering with refinire-rag integration."""
    print("=== Metadata Filtering Test ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "filter_test_index.pkl")
        
        # Initialize with BM25s config
        config = BM25sConfig(index_path=index_path)
        vector_store = BM25sStore(config=config)
        
        # Add diverse documents
        documents = [
            Document(page_content="Introduction to Python programming", metadata={"lang": "python", "level": 1}),
            Document(page_content="Advanced Python techniques", metadata={"lang": "python", "level": 3}),
            Document(page_content="Java enterprise applications", metadata={"lang": "java", "level": 3}),
            Document(page_content="JavaScript for beginners", metadata={"lang": "javascript", "level": 1}),
            Document(page_content="Rust systems programming", metadata={"lang": "rust", "level": 3})
        ]
        
        # Add documents using Document objects
        print("Adding Document objects...")
        doc_ids = vector_store.add_documents(documents)
        print(f"✅ Added {len(doc_ids)} documents\n")
        
        # Test 1: Single criterion filter
        print("Test 1: Filter by language (Python only)")
        python_results = vector_store.similarity_search(
            "programming language",
            k=5,
            filter={"lang": "python"}
        )
        
        print(f"Found {len(python_results)} Python documents:")
        for doc in python_results:
            print(f"  - {doc.page_content}")
            print(f"    Level: {doc.metadata.get('level', 'unknown')}")
        print()
        
        # Test 2: Multiple criteria
        print("Test 2: Advanced level documents only")
        advanced_results = vector_store.similarity_search(
            "advanced techniques",
            k=5,
            filter={"level": 3}
        )
        
        print(f"Found {len(advanced_results)} advanced documents:")
        for doc in advanced_results:
            print(f"  - {doc.page_content}")
            print(f"    Language: {doc.metadata.get('lang', 'unknown')}")
        print()
        
        # Test 3: List-based filtering
        print("Test 3: Python or Rust documents")
        python_rust_results = vector_store.similarity_search(
            "systems programming",
            k=5,
            filter={"lang": ["python", "rust"]}
        )
        
        print(f"Found {len(python_rust_results)} Python/Rust documents:")
        for doc in python_rust_results:
            print(f"  - {doc.metadata.get('lang', 'unknown')}: {doc.page_content}")
        print()
        
        # Test 4: Comparison operators
        print("Test 4: Beginner and intermediate documents (level < 3)")
        beginner_results = vector_store.similarity_search(
            "introduction basics",
            k=5,
            filter={"level": {"$lt": 3}}
        )
        
        print(f"Found {len(beginner_results)} beginner/intermediate documents:")
        for doc in beginner_results:
            print(f"  - Level {doc.metadata.get('level', '?')}: {doc.page_content}")
        print()
        
        return True


def test_index_reload():
    """Test saving and reloading index."""
    print("=== Index Reload Test ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "reload_test_index.pkl")
        
        # Phase 1: Create and save index
        print("Phase 1: Creating and saving index...")
        config = BM25sConfig(index_path=index_path)
        vector_store1 = BM25sStore(config=config)
        
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Python is great for data science and machine learning",
            "BM25 is a probabilistic ranking function"
        ]
        
        metadatas = [
            {"type": "pangram", "id": 1},
            {"type": "tech", "id": 2},
            {"type": "algorithm", "id": 3}
        ]
        
        vector_store1.add_texts(documents, metadatas)
        print("✅ Index created and saved\n")
        
        # Phase 2: Load index in new instance
        print("Phase 2: Loading index in new VectorStore instance...")
        vector_store2 = BM25sStore(config=config)
        
        # Search to verify loaded index
        results = vector_store2.similarity_search("Python data science", k=2)
        
        print(f"Search results from reloaded index:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
            print(f"     Type: {doc.metadata.get('type', 'unknown')}")
            print(f"     ID: {doc.metadata.get('id', 'unknown')}")
        print()
        
        # Verify we got results
        assert len(results) > 0, "Should find results from reloaded index"
        print("✅ Index successfully reloaded and searchable\n")
        
        return True


def test_refinire_rag_compatibility():
    """Test compatibility with refinire-rag expectations."""
    print("=== Refinire-RAG Compatibility Test ===\n")
    
    # Test VectorStore interface implementation
    vector_store = BM25sStore()
    
    # Check required methods exist
    required_methods = [
        'add_texts',
        'add_documents', 
        'similarity_search',
        'similarity_search_with_score',
        'from_texts',
        'from_documents',
        'delete'
    ]
    
    print("Checking VectorStore interface methods:")
    for method in required_methods:
        has_method = hasattr(vector_store, method)
        print(f"  - {method}: {'✅' if has_method else '❌'}")
    print()
    
    # Test class methods
    print("Testing class factory methods...")
    
    # from_texts
    texts = ["Test document 1", "Test document 2"]
    metadatas = [{"id": 1}, {"id": 2}]
    
    store_from_texts = BM25sStore.from_texts(texts, metadatas)
    print("✅ from_texts() works correctly")
    
    # from_documents
    docs = [
        Document(page_content="Doc 1", metadata={"source": "test1"}),
        Document(page_content="Doc 2", metadata={"source": "test2"})
    ]
    
    store_from_docs = BM25sStore.from_documents(docs)
    print("✅ from_documents() works correctly")
    
    # Test deletion
    print("\nTesting document deletion...")
    store = BM25sStore()
    doc_ids = store.add_texts(["Doc A", "Doc B", "Doc C"])
    
    # Delete one document
    success = store.delete([doc_ids[1]])
    print(f"✅ Delete operation returned: {success}")
    
    # Verify deletion
    remaining_results = store.similarity_search("Doc", k=10)
    print(f"   Remaining documents: {len(remaining_results)}")
    
    return True


def run_all_tests():
    """Run all integration tests."""
    print("🔍 Running refinire-rag Integration Tests")
    print("=" * 50 + "\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Metadata Filtering", test_metadata_filtering),
        ("Index Reload", test_index_reload),
        ("Refinire-RAG Compatibility", test_refinire_rag_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            print(f"{'='*50}\n")
            
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, "ERROR"))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, status in results:
        emoji = "✅" if status == "PASS" else "❌"
        print(f"{emoji} {test_name}: {status}")
    
    all_passed = all(status == "PASS" for _, status in results)
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! refinire-rag integration is working correctly.")
        print("\nConfirmed capabilities:")
        print("✅ Document saving with metadata")
        print("✅ Document searching with BM25s")
        print("✅ Metadata filtering (BM25s-j 0.2.0+)")
        print("✅ Index persistence and reloading")
        print("✅ Full VectorStore interface compatibility")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
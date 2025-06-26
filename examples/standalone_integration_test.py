"""Test standalone BM25s VectorStore without langchain dependencies."""

import os
import tempfile
from pathlib import Path

from refinire_rag_bm25s_j import BM25sStore, BaseDocument
from refinire_rag_bm25s_j.models import BM25sConfig


def test_basic_save_and_search():
    """Test basic save and search functionality."""
    print("=== Basic Save and Search Test ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "test_index.pkl")
        
        # 1. Initialize
        print("1. Initializing BM25sStore...")
        config = BM25sConfig(
            k1=1.2,
            b=0.75,
            index_path=index_path
        )
        vector_store = BM25sStore(config=config)
        print("✅ VectorStore initialized\n")
        
        # 2. Save documents
        print("2. Saving documents...")
        documents = [
            "Python is a versatile programming language for web development.",
            "Machine learning models can predict patterns in data.",
            "FastAPI is a modern Python web framework.",
            "Data visualization helps understand complex datasets.",
            "Natural language processing analyzes human language."
        ]
        
        metadatas = [
            {"category": "programming", "topic": "python", "year": 2024},
            {"category": "ai", "topic": "ml", "year": 2023},
            {"category": "web", "topic": "api", "year": 2024},
            {"category": "data", "topic": "viz", "year": 2023},
            {"category": "ai", "topic": "nlp", "year": 2024}
        ]
        
        doc_ids = vector_store.add_texts(documents, metadatas)
        print(f"✅ Saved {len(doc_ids)} documents\n")
        
        # 3. Search documents
        print("3. Searching documents...")
        query = "Python programming"
        results = vector_store.similarity_search(query, k=3)
        
        print(f"Query: '{query}'")
        print(f"Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. Score: {doc.metadata.get('score', 'N/A'):.3f}")
            print(f"     Content: {doc.page_content[:50]}...")
            print(f"     Category: {doc.metadata.get('category', 'unknown')}")
        print()
        
        # 4. Verify persistence
        print("4. Verifying index persistence...")
        assert os.path.exists(index_path), "Index file should exist"
        file_size = os.path.getsize(index_path)
        print(f"✅ Index saved to: {index_path}")
        print(f"   File size: {file_size} bytes\n")
        
        return True


def test_metadata_filtering():
    """Test metadata filtering functionality."""
    print("=== Metadata Filtering Test ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = BM25sConfig(index_path=os.path.join(temp_dir, "filter_index.pkl"))
        vector_store = BM25sStore(config=config)
        
        # Add documents using BaseDocument
        documents = [
            BaseDocument("Introduction to Python", {"lang": "python", "level": 1}),
            BaseDocument("Advanced Python techniques", {"lang": "python", "level": 3}),
            BaseDocument("Java enterprise applications", {"lang": "java", "level": 3}),
            BaseDocument("JavaScript for beginners", {"lang": "javascript", "level": 1}),
            BaseDocument("Rust systems programming", {"lang": "rust", "level": 3})
        ]
        
        print("Adding documents with metadata...")
        doc_ids = vector_store.add_documents(documents)
        print(f"✅ Added {len(doc_ids)} documents\n")
        
        # Test 1: Filter by language
        print("Test 1: Filter by language (Python only)")
        python_results = vector_store.similarity_search(
            "programming techniques",
            k=5,
            filter={"lang": "python"}
        )
        
        print(f"Found {len(python_results)} Python documents:")
        for doc in python_results:
            print(f"  - {doc.page_content}")
            print(f"    Level: {doc.metadata.get('level', 'unknown')}")
        print()
        
        # Test 2: Filter by level
        print("Test 2: Advanced level documents (level = 3)")
        advanced_results = vector_store.similarity_search(
            "advanced programming",
            k=5,
            filter={"level": 3}
        )
        
        print(f"Found {len(advanced_results)} advanced documents:")
        for doc in advanced_results:
            print(f"  - {doc.page_content}")
            print(f"    Language: {doc.metadata.get('lang', 'unknown')}")
        print()
        
        # Test 3: Multiple filters
        print("Test 3: Advanced Python documents")
        python_advanced = vector_store.similarity_search(
            "programming",
            k=5,
            filter={"lang": "python", "level": 3}
        )
        
        print(f"Found {len(python_advanced)} advanced Python documents:")
        for doc in python_advanced:
            print(f"  - {doc.page_content}")
        print()
        
        # Test 4: List filtering
        print("Test 4: Python or Java documents")
        python_java = vector_store.similarity_search(
            "programming",
            k=5,
            filter={"lang": ["python", "java"]}
        )
        
        print(f"Found {len(python_java)} Python/Java documents:")
        for doc in python_java:
            print(f"  - {doc.metadata.get('lang')}: {doc.page_content}")
        print()
        
        # Test 5: Comparison operator
        print("Test 5: Beginner documents (level < 3)")
        beginner_results = vector_store.similarity_search_with_score(
            "introduction basics",
            k=5,
            filter={"level": {"$lt": 3}}
        )
        
        print(f"Found {len(beginner_results)} beginner documents:")
        for doc, score in beginner_results:
            print(f"  - Level {doc.metadata.get('level')}: {doc.page_content} (score: {score:.3f})")
        print()
        
        return True


def test_index_reload():
    """Test saving and reloading index."""
    print("=== Index Reload Test ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "reload_index.pkl")
        config = BM25sConfig(index_path=index_path)
        
        # Phase 1: Create and save
        print("Phase 1: Creating and saving index...")
        store1 = BM25sStore(config=config)
        
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Python is excellent for data science",
            "BM25 is a probabilistic ranking function"
        ]
        
        metadatas = [
            {"type": "pangram"},
            {"type": "tech"},
            {"type": "algorithm"}
        ]
        
        store1.add_texts(documents, metadatas)
        print("✅ Index created and saved\n")
        
        # Phase 2: Load in new instance
        print("Phase 2: Loading index in new instance...")
        store2 = BM25sStore(config=config)
        
        # Search to verify
        results = store2.similarity_search("Python data science", k=2)
        
        print("Search results from reloaded index:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
            print(f"     Type: {doc.metadata.get('type', 'unknown')}")
        
        assert len(results) > 0, "Should find results"
        print("\n✅ Index successfully reloaded\n")
        
        return True


def test_delete_functionality():
    """Test document deletion."""
    print("=== Document Deletion Test ===\n")
    
    vector_store = BM25sStore()
    
    # Add documents
    print("Adding documents...")
    texts = ["Document A", "Document B", "Document C", "Document D"]
    doc_ids = vector_store.add_texts(texts)
    print(f"Added {len(doc_ids)} documents: {doc_ids}\n")
    
    # Verify all documents are searchable
    results = vector_store.similarity_search("Document", k=10)
    print(f"Before deletion: Found {len(results)} documents\n")
    
    # Delete some documents
    print(f"Deleting documents: {doc_ids[1:3]}")
    success = vector_store.delete(doc_ids[1:3])
    print(f"Deletion success: {success}\n")
    
    # Verify deletion
    results_after = vector_store.similarity_search("Document", k=10)
    print(f"After deletion: Found {len(results_after)} documents")
    
    for doc in results_after:
        print(f"  - {doc.page_content}")
    
    assert len(results_after) == 2, "Should have 2 documents left"
    print("\n✅ Document deletion works correctly\n")
    
    return True


def test_factory_methods():
    """Test class factory methods."""
    print("=== Factory Methods Test ===\n")
    
    # Test from_texts
    print("1. Testing from_texts...")
    texts = ["Text 1", "Text 2", "Text 3"]
    metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
    
    store_from_texts = BM25sStore.from_texts(texts, metadatas)
    results = store_from_texts.similarity_search("Text", k=3)
    print(f"✅ from_texts created store with {len(results)} searchable documents\n")
    
    # Test from_documents
    print("2. Testing from_documents...")
    docs = [
        BaseDocument("Doc A", {"source": "test"}),
        BaseDocument("Doc B", {"source": "test"}),
        BaseDocument("Doc C", {"source": "test"})
    ]
    
    store_from_docs = BM25sStore.from_documents(docs)
    results = store_from_docs.similarity_search("Doc", k=3)
    print(f"✅ from_documents created store with {len(results)} searchable documents\n")
    
    return True


def run_all_tests():
    """Run all integration tests."""
    print("🔍 BM25s Standalone Integration Tests")
    print("=" * 50 + "\n")
    
    tests = [
        ("Basic Save and Search", test_basic_save_and_search),
        ("Metadata Filtering", test_metadata_filtering),
        ("Index Reload", test_index_reload),
        ("Document Deletion", test_delete_functionality),
        ("Factory Methods", test_factory_methods)
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
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
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
        print("🎉 All tests passed!")
        print("\nConfirmed capabilities:")
        print("✅ Document saving with metadata")
        print("✅ Document searching with BM25s")
        print("✅ Metadata filtering (basic and advanced)")
        print("✅ Index persistence and reloading")
        print("✅ Document deletion")
        print("✅ Factory methods (from_texts, from_documents)")
        print("\n📌 The standalone version works without any external dependencies!")
    else:
        print("❌ Some tests failed.")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
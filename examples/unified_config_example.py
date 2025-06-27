"""Example: Unified Configuration Approach for BM25s Plugin."""

import os
import tempfile
from refinire_rag_bm25s_j.keyword_store import BM25sKeywordStore
from refinire_rag.models.document import Document


def main():
    """Demonstrate unified configuration approach for BM25s plugin."""
    
    print("🔧 BM25s Unified Configuration Example")
    print("=" * 45)
    
    # Method 1: Using kwargs (highest priority)
    print("\n1️⃣ Configuration via kwargs:")
    store1 = BM25sKeywordStore(
        k1=1.5,
        b=0.8,
        method="bm25+",
        stemmer="janome",
        index_path="./data/kwargs_index.pkl"
    )
    config1 = store1.get_config()
    print(f"   k1: {config1['k1']}")
    print(f"   b: {config1['b']}")
    print(f"   method: {config1['method']}")
    print(f"   stemmer: {config1['stemmer']}")
    
    # Method 2: Using environment variables
    print("\n2️⃣ Configuration via environment variables:")
    os.environ["REFINIRE_RAG_BM25S_K1"] = "2.0"
    os.environ["REFINIRE_RAG_BM25S_B"] = "0.9"
    os.environ["REFINIRE_RAG_BM25S_METHOD"] = "bm25l"
    os.environ["REFINIRE_RAG_BM25S_INDEX_PATH"] = "./data/env_index.pkl"
    
    store2 = BM25sKeywordStore()
    config2 = store2.get_config()
    print(f"   k1: {config2['k1']} (from env)")
    print(f"   b: {config2['b']} (from env)")
    print(f"   method: {config2['method']} (from env)")
    print(f"   index_path: {config2['index_path']} (from env)")
    
    # Method 3: Priority demonstration (kwargs > env vars > defaults)
    print("\n3️⃣ Configuration priority (kwargs > env vars > defaults):")
    store3 = BM25sKeywordStore(
        k1=3.0,  # This overrides env var
        epsilon=0.5  # This uses kwargs value
        # b and method will use env vars
        # stemmer and stopwords will use defaults
    )
    config3 = store3.get_config()
    print(f"   k1: {config3['k1']} (kwargs override)")
    print(f"   b: {config3['b']} (from env)")
    print(f"   epsilon: {config3['epsilon']} (from kwargs)")
    print(f"   method: {config3['method']} (from env)")
    print(f"   stemmer: {config3['stemmer']} (default)")
    print(f"   stopwords: {config3['stopwords']} (default)")
    
    # Method 4: Legacy config parameter (backward compatibility)
    print("\n4️⃣ Legacy config parameter (deprecated but supported):")
    legacy_config = {
        "k1": 1.8,
        "b": 0.85,
        "index_path": "./data/legacy_index.pkl"
    }
    store4 = BM25sKeywordStore(config=legacy_config)
    config4 = store4.get_config()
    print(f"   k1: {config4['k1']} (legacy config)")
    print(f"   b: {config4['b']} (legacy config)")
    print(f"   index_path: {config4['index_path']} (legacy config)")
    
    # Demonstration with actual documents
    print("\n📚 Testing with actual documents:")
    temp_dir = tempfile.mkdtemp()
    index_path = os.path.join(temp_dir, "unified_test.pkl")
    
    # Use kwargs for clean configuration
    test_store = BM25sKeywordStore(
        k1=1.2,
        b=0.75,
        index_path=index_path,
        method="bm25",
        stemmer="janome"
    )
    
    # Add sample documents
    documents = [
        Document(
            id="doc1",
            content="機械学習は人工知能の重要な技術分野です。",
            metadata={"category": "AI"}
        ),
        Document(
            id="doc2",
            content="自然言語処理はテキスト解析に使用されます。",
            metadata={"category": "NLP"}
        )
    ]
    
    test_store.index_documents(documents)
    print(f"   Indexed {test_store.get_document_count()} documents")
    
    # Search
    results = test_store.search("機械学習", limit=2)
    print(f"   Search results: {len(results)} found")
    for result in results:
        print(f"     - {result.document.id}: score={result.score:.3f}")
    
    # Clean up environment variables
    print("\n🧹 Cleaning up environment variables:")
    env_vars_to_remove = [
        "REFINIRE_RAG_BM25S_K1",
        "REFINIRE_RAG_BM25S_B", 
        "REFINIRE_RAG_BM25S_METHOD",
        "REFINIRE_RAG_BM25S_INDEX_PATH"
    ]
    
    for env_var in env_vars_to_remove:
        if env_var in os.environ:
            del os.environ[env_var]
            print(f"   Removed: {env_var}")
    
    # Clean up temp files
    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        os.rmdir(temp_dir)
        print("   Removed temporary files")
    except Exception as e:
        print(f"   Cleanup warning: {e}")
    
    print("\n✅ Configuration examples completed!")
    print("\n💡 Recommendations:")
    print("   - Use kwargs for programmatic configuration")
    print("   - Use environment variables for deployment configuration") 
    print("   - Combine both for flexible application settings")
    print("   - Environment variable format: REFINIRE_RAG_BM25S_<SETTING>")


if __name__ == "__main__":
    main()
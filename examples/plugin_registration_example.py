"""Example: Using BM25sKeywordStore as a registered plugin with refinire-rag."""

import os
import tempfile
from typing import Dict, Any

from refinire_rag.models.document import Document


def demonstrate_plugin_discovery():
    """Demonstrate how BM25s plugin is discovered and loaded."""
    
    print("🔌 BM25s Plugin Registration & Discovery Example")
    print("=" * 55)
    
    # This would normally be done by refinire-rag's plugin discovery system
    print("\n📋 Plugin Registration Information:")
    print("-" * 40)
    
    plugin_info = {
        "name": "bm25s_keyword",
        "entry_point": "refinire_rag_bm25s_j.keyword_store:BM25sKeywordStore",
        "type": "keyword_store",
        "description": "BM25s-based keyword search with Japanese text support"
    }
    
    for key, value in plugin_info.items():
        print(f"  {key}: {value}")
    
    # Environment template information
    print("\n⚙️ Environment Configuration Template:")
    print("-" * 45)
    
    try:
        from refinire_rag_bm25s_j.env_template import bm25s_keyword_env_template
        
        env_template = bm25s_keyword_env_template()
        
        for env_var, config in env_template.items():
            print(f"\n  {env_var}:")
            print(f"    Description: {config['description']}")
            print(f"    Type: {config['type']}")
            print(f"    Default: {config['default']}")
            print(f"    Required: {config['required']}")
    
    except ImportError as e:
        print(f"  ⚠️ Could not load environment template: {e}")
    
    print("\n🏗️ Plugin Usage Examples:")
    print("-" * 30)
    
    # Example 1: Direct instantiation (development/testing)
    print("\n1️⃣ Direct instantiation:")
    print("```python")
    print("from refinire_rag_bm25s_j.keyword_store import BM25sKeywordStore")
    print("")
    print("config = {")
    print("    'index_path': 'data/my_index.pkl',")
    print("    'k1': 1.2,")
    print("    'b': 0.75,")
    print("    'epsilon': 0.25")
    print("}")
    print("")
    print("keyword_store = BM25sKeywordStore(config=config)")
    print("```")
    
    # Example 2: Plugin discovery (production)
    print("\n2️⃣ Via plugin discovery (when integrated with refinire-rag):")
    print("```python")
    print("# This would be done by refinire-rag's plugin system")
    print("from refinire_rag.factories import create_keyword_store")
    print("")
    print("keyword_store = create_keyword_store(")
    print("    'bm25s_keyword',")
    print("    config={")
    print("        'index_path': 'data/my_index.pkl',")
    print("        'k1': 1.2")
    print("    }")
    print(")")
    print("```")
    
    # Example 3: Environment-based configuration
    print("\n3️⃣ Environment-based configuration:")
    print("```bash")
    print("export BM25S_INDEX_PATH=/path/to/index.pkl")
    print("export BM25S_K1=1.5")
    print("export BM25S_B=0.8")
    print("export BM25S_STEMMER=janome")
    print("```")
    
    print("\n```python")
    print("# Configuration automatically loaded from environment")
    print("keyword_store = create_keyword_store('bm25s_keyword')")
    print("```")


def demonstrate_practical_usage():
    """Demonstrate practical usage patterns."""
    
    print("\n\n💼 Practical Usage Patterns")
    print("=" * 35)
    
    # Setup
    temp_dir = tempfile.mkdtemp()
    index_path = os.path.join(temp_dir, "plugin_demo_index.pkl")
    
    try:
        from refinire_rag_bm25s_j.keyword_store import BM25sKeywordStore
        
        # Configuration patterns
        print("\n⚙️ Configuration Patterns:")
        print("-" * 30)
        
        # Pattern 1: Basic configuration
        basic_config = {
            "index_path": index_path
        }
        
        print("\n1️⃣ Basic configuration (using defaults):")
        keyword_store_basic = BM25sKeywordStore(config=basic_config)
        print(f"   ✅ Created with defaults")
        print(f"   📊 K1: {keyword_store_basic.bm25s_config.k1}")
        print(f"   📊 B: {keyword_store_basic.bm25s_config.b}")
        print(f"   📊 Epsilon: {keyword_store_basic.bm25s_config.epsilon}")
        
        # Pattern 2: Custom tuning
        tuned_config = {
            "index_path": index_path,
            "k1": 1.5,      # Higher term frequency sensitivity
            "b": 0.8,       # More length normalization
            "epsilon": 0.1  # Lower IDF cutoff
        }
        
        print("\n2️⃣ Performance-tuned configuration:")
        keyword_store_tuned = BM25sKeywordStore(config=tuned_config)
        print(f"   ✅ Created with custom parameters")
        print(f"   📊 K1: {keyword_store_tuned.bm25s_config.k1} (higher sensitivity)")
        print(f"   📊 B: {keyword_store_tuned.bm25s_config.b} (more normalization)")
        print(f"   📊 Epsilon: {keyword_store_tuned.bm25s_config.epsilon} (lower cutoff)")
        
        # Usage demonstration
        print("\n📝 Document Processing Demo:")
        print("-" * 35)
        
        sample_docs = [
            Document(
                id="plugin_doc_1",
                content="BM25sプラグインは日本語テキストの検索に最適化されています。形態素解析により高精度な検索が可能です。",
                metadata={"type": "plugin_info", "language": "ja"}
            ),
            Document(
                id="plugin_doc_2", 
                content="Plugin architecture allows seamless integration with refinire-rag framework. Multiple keyword stores can be used simultaneously.",
                metadata={"type": "architecture", "language": "en"}
            ),
            Document(
                id="plugin_doc_3",
                content="設定可能なパラメータにより、様々なドメインやテキストタイプに対応できます。カスタマイズ性が高いのが特徴です。",
                metadata={"type": "configuration", "language": "ja"}
            )
        ]
        
        # Process documents
        print(f"\n🔄 Processing {len(sample_docs)} documents...")
        processed = keyword_store_basic.process(sample_docs)
        print(f"   ✅ Processed {len(processed)} documents")
        print(f"   📊 Index size: {keyword_store_basic.get_document_count()}")
        
        # Search demonstration
        search_queries = [
            ("プラグイン", "Japanese plugin search"),
            ("Plugin", "English plugin search"), 
            ("設定", "Configuration search"),
            ("framework", "Framework search")
        ]
        
        print("\n🔍 Multi-language Search Demo:")
        print("-" * 35)
        
        for query, description in search_queries:
            print(f"\n🔎 {description}: '{query}'")
            results = keyword_store_basic.search(query, limit=2)
            
            for i, result in enumerate(results, 1):
                doc = result.document
                lang = doc.metadata.get('language', 'unknown')
                doc_type = doc.metadata.get('type', 'unknown')
                print(f"   {i}. [{result.score:.3f}] {doc.id}")
                print(f"      Language: {lang} | Type: {doc_type}")
                print(f"      Content: {doc.content[:60]}...")
        
        # Performance comparison
        print("\n⚡ Configuration Performance Comparison:")
        print("-" * 45)
        
        test_query = "検索"
        
        print(f"\n🔎 Query: '{test_query}'")
        
        # Basic config results
        basic_results = keyword_store_basic.search(test_query, limit=3)
        print(f"\n📊 Basic config results ({len(basic_results)} found):")
        for result in basic_results:
            print(f"   [{result.score:.4f}] {result.document.id}")
        
        # Tuned config results
        # Add same documents to tuned store for comparison
        keyword_store_tuned.process(sample_docs)
        tuned_results = keyword_store_tuned.search(test_query, limit=3)
        print(f"\n📊 Tuned config results ({len(tuned_results)} found):")
        for result in tuned_results:
            print(f"   [{result.score:.4f}] {result.document.id}")
        
        print("\n✅ Plugin demonstration completed!")
        
    finally:
        # Cleanup
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


def main():
    """Run the complete plugin registration example."""
    
    # Part 1: Plugin discovery and registration info
    demonstrate_plugin_discovery()
    
    # Part 2: Practical usage patterns
    demonstrate_practical_usage()
    
    print("\n" + "=" * 60)
    print("🎉 BM25s Plugin Registration Example Complete!")
    print("")
    print("📝 Key Takeaways:")
    print("   • BM25s is registered as 'bm25s_keyword' plugin")
    print("   • Supports environment-based configuration")
    print("   • Optimized for Japanese text processing")
    print("   • Seamlessly integrates with refinire-rag framework")
    print("   • Configurable parameters for domain adaptation")
    print("=" * 60)


if __name__ == "__main__":
    main()
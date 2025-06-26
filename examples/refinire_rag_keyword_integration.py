"""Example: Integrating BM25sKeywordStore with refinire-rag framework."""

import os
import tempfile
from typing import List

from refinire_rag import CorpusManager, DocumentProcessor, QueryEngine
from refinire_rag.models.document import Document
from refinire_rag_bm25s_j.keyword_store import BM25sKeywordStore


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            id="tech_001",
            content="人工知能（AI）技術の発展により、自動運転車の実現が近づいています。機械学習アルゴリズムが大量のセンサーデータを処理し、安全な運転判断を行います。",
            metadata={
                "category": "technology",
                "topic": "AI",
                "industry": "automotive",
                "date": "2024-01-15"
            }
        ),
        Document(
            id="tech_002", 
            content="自然言語処理（NLP）の最新技術により、チャットボットの対話品質が大幅に向上しました。深層学習モデルを使用することで、より自然な会話が可能になります。",
            metadata={
                "category": "technology",
                "topic": "NLP", 
                "industry": "software",
                "date": "2024-01-16"
            }
        ),
        Document(
            id="business_001",
            content="デジタルトランスフォーメーション（DX）が企業経営において重要な戦略となっています。クラウドコンピューティングやビッグデータ分析を活用し、業務効率化を図ります。",
            metadata={
                "category": "business",
                "topic": "DX",
                "industry": "consulting", 
                "date": "2024-01-17"
            }
        ),
        Document(
            id="research_001",
            content="量子コンピュータの研究開発が急速に進展しています。従来のコンピュータでは解決困難な最適化問題や暗号解読に革新をもたらす可能性があります。",
            metadata={
                "category": "research",
                "topic": "quantum",
                "industry": "science",
                "date": "2024-01-18"
            }
        ),
        Document(
            id="health_001",
            content="医療分野でのAI活用が拡大しています。医療画像の診断支援や薬剤開発の効率化により、より精密で迅速な医療サービスの提供が期待されます。",
            metadata={
                "category": "healthcare",
                "topic": "medical_AI",
                "industry": "healthcare",
                "date": "2024-01-19"
            }
        )
    ]


def main():
    """Demonstrate BM25sKeywordStore integration with refinire-rag."""
    
    print("🔗 BM25s KeywordStore + refinire-rag Integration Example")
    print("=" * 60)
    
    # Setup temporary directory
    temp_dir = tempfile.mkdtemp()
    index_path = os.path.join(temp_dir, "integrated_bm25s_index.pkl")
    
    try:
        # 1. Initialize BM25sKeywordStore as DocumentProcessor
        print("\n📚 Setting up BM25s KeywordStore...")
        
        keyword_config = {
            "index_path": index_path,
            "k1": 1.2,
            "b": 0.75,
            "epsilon": 0.25
        }
        
        bm25s_processor = BM25sKeywordStore(config=keyword_config)
        print(f"✅ BM25sKeywordStore initialized with config: {keyword_config}")
        
        # 2. Create sample documents
        documents = create_sample_documents()
        print(f"\n📝 Created {len(documents)} sample documents")
        
        # 3. Use BM25sKeywordStore as DocumentProcessor
        print("\n🔄 Processing documents through BM25s pipeline...")
        processed_docs = bm25s_processor.process(documents)
        print(f"✅ Processed {len(processed_docs)} documents")
        print(f"📊 Index contains {bm25s_processor.get_document_count()} documents")
        
        # 4. Demonstrate keyword search capabilities
        print("\n🔍 Keyword Search Demonstrations:")
        print("-" * 40)
        
        search_examples = [
            {
                "query": "人工知能",
                "description": "AI technology search"
            },
            {
                "query": "コンピュータ",
                "description": "Computer-related content"
            },
            {
                "query": "医療",
                "description": "Healthcare domain"
            },
            {
                "query": "効率化",
                "description": "Efficiency improvements"
            }
        ]
        
        for example in search_examples:
            query = example["query"]
            desc = example["description"]
            
            print(f"\n🔎 {desc} - Query: '{query}'")
            results = bm25s_processor.search(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    doc = result.document
                    print(f"  {i}. [{result.score:.3f}] {doc.id}")
                    print(f"     Category: {doc.metadata.get('category', 'N/A')}")
                    print(f"     Topic: {doc.metadata.get('topic', 'N/A')}")
                    print(f"     Content: {doc.content[:70]}...")
            else:
                print("  No results found")
        
        # 5. Demonstrate metadata filtering
        print("\n🎯 Metadata Filtering Examples:")
        print("-" * 35)
        
        filtering_examples = [
            {
                "query": "技術",
                "filter": {"category": "technology"},
                "description": "Technology category only"
            },
            {
                "query": "AI",
                "filter": {"industry": "automotive"},
                "description": "Automotive industry only"
            },
            {
                "query": "コンピュータ",
                "filter": {"topic": "quantum"},
                "description": "Quantum computing topic"
            }
        ]
        
        for example in filtering_examples:
            query = example["query"]
            metadata_filter = example["filter"]
            desc = example["description"]
            
            print(f"\n🔎 {desc}")
            print(f"   Query: '{query}' | Filter: {metadata_filter}")
            
            filtered_results = bm25s_processor.retrieve(
                query,
                limit=5,
                metadata_filter=metadata_filter
            )
            
            if filtered_results:
                for i, result in enumerate(filtered_results, 1):
                    doc = result.document
                    print(f"  {i}. [{result.score:.3f}] {doc.id}")
                    filter_values = {k: doc.metadata.get(k, 'N/A') for k in metadata_filter.keys()}
                    print(f"     Matched: {filter_values}")
                    print(f"     Content: {doc.content[:60]}...")
            else:
                print("  No results found with this filter")
        
        # 6. Demonstrate document management in processing context
        print("\n📝 Dynamic Document Management:")
        print("-" * 40)
        
        # Add new document during processing
        new_doc = Document(
            id="sustainability_001",
            content="持続可能な社会の実現に向けて、再生可能エネルギーとスマートグリッド技術の統合が進んでいます。AIによる電力需要予測と最適化により、効率的なエネルギー管理が可能になります。",
            metadata={
                "category": "sustainability",
                "topic": "renewable_energy",
                "industry": "energy",
                "date": "2024-01-20"
            }
        )
        
        print(f"\n➕ Adding new document: {new_doc.id}")
        bm25s_processor.add_document(new_doc)
        print(f"✅ Total documents: {bm25s_processor.get_document_count()}")
        
        # Search for new content
        print("\n🔎 Searching for newly added content: '持続可能'")
        sustainability_results = bm25s_processor.search("持続可能", limit=2)
        for result in sustainability_results:
            print(f"  Found: {result.document.id} [Score: {result.score:.3f}]")
            print(f"  Category: {result.document.metadata.get('category', 'N/A')}")
        
        # 7. Demonstrate batch processing workflow
        print("\n🔄 Batch Processing Workflow:")
        print("-" * 35)
        
        batch_documents = [
            Document(
                id="iot_001",
                content="IoT（モノのインターネット）デバイスの普及により、スマートシティの構築が現実的になっています。センサーネットワークからのリアルタイムデータを活用し、都市運営を最適化します。",
                metadata={"category": "technology", "topic": "IoT", "industry": "urban_planning"}
            ),
            Document(
                id="security_001", 
                content="サイバーセキュリティの脅威が高度化する中、AIベースの異常検知システムの導入が急務となっています。機械学習により、未知の攻撃パターンも検出可能です。",
                metadata={"category": "security", "topic": "cybersecurity", "industry": "IT"}
            )
        ]
        
        print(f"🔄 Processing batch of {len(batch_documents)} documents...")
        batch_processed = bm25s_processor.process(batch_documents)
        print(f"✅ Batch processed {len(batch_processed)} documents")
        print(f"📊 Final index size: {bm25s_processor.get_document_count()} documents")
        
        # 8. Advanced search demonstration
        print("\n🚀 Advanced Search Features:")
        print("-" * 35)
        
        # Complex query with multiple terms
        complex_query = "AI 技術 活用"
        print(f"\n🔎 Complex query: '{complex_query}'")
        complex_results = bm25s_processor.search(complex_query, limit=5)
        
        for i, result in enumerate(complex_results, 1):
            doc = result.document
            print(f"  {i}. [{result.score:.3f}] {doc.id}")
            print(f"     Category: {doc.metadata.get('category', 'N/A')}")
            print(f"     Industry: {doc.metadata.get('industry', 'N/A')}")
            print(f"     Relevance: {doc.content[:80]}...")
        
        # Multi-constraint filtering
        print(f"\n🎯 Multi-constraint search: 'システム' in technology category")
        multi_filter_results = bm25s_processor.retrieve(
            "システム",
            limit=3,
            metadata_filter={"category": "technology"}
        )
        
        for result in multi_filter_results:
            doc = result.document
            print(f"  [{result.score:.3f}] {doc.id} - {doc.metadata.get('topic', 'N/A')}")
        
        # 9. Performance and statistics
        print("\n📊 Search Performance Statistics:")
        print("-" * 40)
        
        total_docs = bm25s_processor.get_document_count()
        print(f"📚 Total indexed documents: {total_docs}")
        
        # Sample queries for performance testing
        performance_queries = ["AI", "技術", "コンピュータ", "システム", "データ"]
        
        for query in performance_queries:
            results = bm25s_processor.search(query, limit=10)
            print(f"🔎 '{query}': {len(results)} results found")
        
        print("\n✅ BM25s KeywordStore integration demonstration completed!")
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
            os.rmdir(temp_dir)
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


if __name__ == "__main__":
    main()
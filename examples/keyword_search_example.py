"""Example: Using BM25sKeywordStore for keyword-based document search."""

import os
import tempfile
from typing import List

from refinire_rag.models.document import Document
from refinire_rag_bm25s_j.keyword_store import BM25sKeywordStore


def main():
    """Demonstrate BM25sKeywordStore functionality."""
    
    # Create temporary directory for index
    temp_dir = tempfile.mkdtemp()
    index_path = os.path.join(temp_dir, "bm25s_index.pkl")
    
    print("🔍 BM25s KeywordSearch Example")
    print("=" * 40)
    
    # Initialize BM25sKeywordStore with unified configuration approach
    # Configuration priority: kwargs > environment variables > defaults
    keyword_store = BM25sKeywordStore(
        index_path=index_path,
        k1=1.2,
        b=0.75,
        epsilon=0.25
    )
    print(f"✅ Initialized BM25sKeywordStore with index: {index_path}")
    
    # Sample Japanese documents
    sample_documents = [
        Document(
            id="doc1",
            content="人工知能とは、機械学習や深層学習などの技術を用いて、人間の知能を模倣するコンピュータシステムです。",
            metadata={"category": "AI", "language": "ja", "author": "研究者A"}
        ),
        Document(
            id="doc2", 
            content="自然言語処理は、コンピュータが人間の言語を理解し、処理する技術分野です。機械翻訳や文書要約などに応用されます。",
            metadata={"category": "NLP", "language": "ja", "author": "研究者B"}
        ),
        Document(
            id="doc3",
            content="機械学習は、データから規則性や法則を自動的に発見し、予測や分類を行う手法です。教師あり学習と教師なし学習があります。",
            metadata={"category": "ML", "language": "ja", "author": "研究者C"}
        ),
        Document(
            id="doc4",
            content="深層学習はニューラルネットワークを用いた機械学習手法で、画像認識や音声認識で高い性能を発揮します。",
            metadata={"category": "DL", "language": "ja", "author": "研究者D"}
        ),
        Document(
            id="doc5",
            content="データサイエンスは、統計学、機械学習、プログラミングを組み合わせてデータから価値のある洞察を得る学問です。",
            metadata={"category": "DS", "language": "ja", "author": "研究者E"}
        )
    ]
    
    # Index documents
    print(f"\n📚 Indexing {len(sample_documents)} documents...")
    keyword_store.index_documents(sample_documents)
    print(f"✅ Successfully indexed {keyword_store.get_document_count()} documents")
    
    # Demonstrate search functionality
    queries = [
        "機械学習",
        "人工知能", 
        "自然言語処理",
        "深層学習",
        "データ"
    ]
    
    print("\n🔍 Search Examples:")
    print("-" * 30)
    
    for query in queries:
        print(f"\n🔎 Query: '{query}'")
        results = keyword_store.search(query, limit=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [Score: {result.score:.3f}] {result.document.id}")
            print(f"     Category: {result.document.metadata.get('category', 'N/A')}")
            print(f"     Content: {result.document.content[:60]}...")
    
    # Demonstrate retrieval with metadata filtering
    print("\n🎯 Metadata Filtering Examples:")
    print("-" * 35)
    
    # Search only in AI category
    print("\n🔎 Query: '学習' (AI category only)")
    ai_results = keyword_store.retrieve(
        "学習",
        limit=5,
        metadata_filter={"category": "AI"}
    )
    
    for i, result in enumerate(ai_results, 1):
        print(f"  {i}. [Score: {result.score:.3f}] {result.document.id}")
        print(f"     Category: {result.document.metadata.get('category', 'N/A')}")
        print(f"     Content: {result.document.content[:60]}...")
    
    # Search only by specific author
    print("\n🔎 Query: 'データ' (by 研究者E only)")
    author_results = keyword_store.retrieve(
        "データ",
        limit=5, 
        metadata_filter={"author": "研究者E"}
    )
    
    for i, result in enumerate(author_results, 1):
        print(f"  {i}. [Score: {result.score:.3f}] {result.document.id}")
        print(f"     Author: {result.document.metadata.get('author', 'N/A')}")
        print(f"     Content: {result.document.content[:60]}...")
    
    # Demonstrate document management
    print("\n📝 Document Management Examples:")
    print("-" * 40)
    
    # Add a new document
    new_doc = Document(
        id="doc6",
        content="ビッグデータとは、従来のデータベース管理システムでは処理が困難な巨大なデータセットのことです。",
        metadata={"category": "BigData", "language": "ja", "author": "研究者F"}
    )
    
    print(f"\n➕ Adding new document: {new_doc.id}")
    keyword_store.add_document(new_doc)
    print(f"✅ Total documents: {keyword_store.get_document_count()}")
    
    # Search for the new content
    print("\n🔎 Searching for 'ビッグデータ':")
    big_data_results = keyword_store.search("ビッグデータ", limit=2)
    for result in big_data_results:
        print(f"  Found: {result.document.id} [Score: {result.score:.3f}]")
    
    # Update a document
    updated_doc = Document(
        id="doc1",
        content="人工知能（AI）とは、機械学習、深層学習、自然言語処理などの技術を統合した総合的なシステムです。現代のAIは様々な分野で活用されています。",
        metadata={"category": "AI", "language": "ja", "author": "研究者A", "updated": True}
    )
    
    print(f"\n✏️ Updating document: {updated_doc.id}")
    update_success = keyword_store.update_document(updated_doc)
    print(f"✅ Update successful: {update_success}")
    
    # Search to verify update
    print("\n🔎 Searching for updated content '総合的':")
    updated_results = keyword_store.search("総合的", limit=1)
    for result in updated_results:
        print(f"  Found: {result.document.id} [Score: {result.score:.3f}]")
        print(f"  Content: {result.document.content[:80]}...")
    
    # Remove a document
    print(f"\n🗑️ Removing document: doc2")
    remove_success = keyword_store.remove_document("doc2")
    print(f"✅ Removal successful: {remove_success}")
    print(f"📊 Total documents after removal: {keyword_store.get_document_count()}")
    
    # Demonstrate processing interface (DocumentProcessor)
    print("\n🔄 DocumentProcessor Interface Example:")
    print("-" * 45)
    
    batch_docs = [
        Document(
            id="batch1",
            content="量子コンピュータは、量子力学の原理を利用した次世代コンピュータです。",
            metadata={"category": "Quantum", "language": "ja"}
        ),
        Document(
            id="batch2", 
            content="ブロックチェーンは、分散型台帳技術として注目されています。",
            metadata={"category": "Blockchain", "language": "ja"}
        )
    ]
    
    print(f"🔄 Processing {len(batch_docs)} documents through pipeline...")
    processed_docs = keyword_store.process(batch_docs)
    print(f"✅ Processed {len(processed_docs)} documents")
    print(f"📊 Total documents: {keyword_store.get_document_count()}")
    
    # Final search demonstration
    print("\n🔍 Final Search Demonstration:")
    print("-" * 35)
    
    final_query = "コンピュータ"
    print(f"🔎 Query: '{final_query}'")
    final_results = keyword_store.search(final_query, limit=5)
    
    for i, result in enumerate(final_results, 1):
        print(f"  {i}. [Score: {result.score:.3f}] {result.document.id}")
        print(f"     Category: {result.document.metadata.get('category', 'N/A')}")
        print(f"     Content: {result.document.content[:60]}...")
    
    # Cleanup
    print(f"\n🧹 Cleaning up temporary files...")
    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        os.rmdir(temp_dir)
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    print("\n🎉 BM25s KeywordSearch Example completed!")


if __name__ == "__main__":
    main()
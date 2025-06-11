"""Hybrid search example combining BM25s with semantic search using refinire-rag."""

from typing import List, Tuple, Dict, Any
import asyncio

try:
    from refinire import Refinire
    from refinire.rag import RAGPipeline
    from refinire.retrievers import HybridRetriever
    from langchain_core.documents import Document
except ImportError:
    print("This example requires refinire-rag. Install it with: uv add refinire-rag")
    exit(1)

from refinire_rag_bm25s_j import BM25sStore
from refinire_rag_bm25s_j.models import BM25sConfig


class BM25sHybridRetriever:
    """Hybrid retriever combining BM25s with semantic search."""
    
    def __init__(
        self,
        bm25s_store: BM25sStore,
        semantic_store=None,  # Would be a semantic vector store
        bm25s_weight: float = 0.7,
        semantic_weight: float = 0.3
    ):
        self.bm25s_store = bm25s_store
        self.semantic_store = semantic_store
        self.bm25s_weight = bm25s_weight
        self.semantic_weight = semantic_weight
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve documents using hybrid approach."""
        
        # Get BM25s results
        bm25s_results = self.bm25s_store.similarity_search_with_score(query, k=k*2)
        
        # Normalize BM25s scores (0-1 range)
        if bm25s_results:
            max_bm25s_score = max(score for _, score in bm25s_results)
            min_bm25s_score = min(score for _, score in bm25s_results)
            score_range = max_bm25s_score - min_bm25s_score or 1
            
            normalized_bm25s = [
                (doc, (score - min_bm25s_score) / score_range * self.bm25s_weight)
                for doc, score in bm25s_results
            ]
        else:
            normalized_bm25s = []
        
        # In a real implementation, you would also get semantic results here
        # semantic_results = self.semantic_store.similarity_search_with_score(query, k=k*2)
        # For this example, we'll simulate semantic results
        semantic_results = self._simulate_semantic_results(query, k*2)
        
        # Combine and rerank results
        combined_results = self._combine_results(normalized_bm25s, semantic_results)
        
        # Return top k results
        return sorted(combined_results, key=lambda x: x[1], reverse=True)[:k]
    
    def _simulate_semantic_results(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Simulate semantic search results for demonstration."""
        # In practice, this would use embeddings and cosine similarity
        # For this example, we'll create mock semantic scores based on query intent
        
        semantic_keywords = {
            "machine learning": ["ai", "deep_learning", "cross_validation"],
            "python": ["programming", "fastapi", "pandas"],
            "data science": ["feature_engineering", "cross_validation"],
            "search": ["bm25", "semantic_search", "vector_databases"]
        }
        
        # Get all documents from BM25s store
        all_docs = self.bm25s_store.similarity_search("", k=20)
        semantic_results = []
        
        for doc in all_docs:
            semantic_score = 0.0
            topic = doc.metadata.get('topic', '')
            
            # Simple semantic scoring based on topic relevance
            for intent, related_topics in semantic_keywords.items():
                if intent.lower() in query.lower():
                    if any(related_topic in topic for related_topic in related_topics):
                        semantic_score += 0.8
                    elif doc.metadata.get('category') in ['ai', 'data_science']:
                        semantic_score += 0.3
            
            if semantic_score > 0:
                # Apply semantic weight
                weighted_score = semantic_score * self.semantic_weight
                semantic_results.append((doc, weighted_score))
        
        return semantic_results[:k]
    
    def _combine_results(
        self,
        bm25s_results: List[Tuple[Document, float]],
        semantic_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Combine BM25s and semantic results."""
        
        # Create a dictionary to combine scores for the same documents
        combined_scores = {}
        
        # Add BM25s scores
        for doc, score in bm25s_results:
            doc_id = doc.metadata.get('topic', id(doc))
            combined_scores[doc_id] = {'doc': doc, 'bm25s': score, 'semantic': 0.0}
        
        # Add semantic scores
        for doc, score in semantic_results:
            doc_id = doc.metadata.get('topic', id(doc))
            if doc_id in combined_scores:
                combined_scores[doc_id]['semantic'] = score
            else:
                combined_scores[doc_id] = {'doc': doc, 'bm25s': 0.0, 'semantic': score}
        
        # Calculate final scores
        final_results = []
        for doc_info in combined_scores.values():
            final_score = doc_info['bm25s'] + doc_info['semantic']
            final_results.append((doc_info['doc'], final_score))
        
        return final_results


def setup_hybrid_rag_system():
    """Set up hybrid RAG system with BM25s and simulated semantic search."""
    
    print("=== Setting up Hybrid RAG System ===\n")
    
    # Configure BM25s
    bm25s_config = BM25sConfig(
        k1=1.2,
        b=0.75,
        epsilon=0.25,
        index_path="./data/hybrid_bm25s_index.pkl"
    )
    
    bm25s_store = BM25sStore(config=bm25s_config)
    
    # Sample technical documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
        "Python is a versatile programming language widely used in data science and web development.",
        "FastAPI is a modern web framework for building APIs with Python, featuring automatic documentation generation.",
        "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        "Pandas provides powerful data structures and operations for manipulating numerical tables and time series.",
        "Natural language processing enables computers to understand and generate human language.",
        "Feature engineering involves creating new variables from existing data to improve model performance.",
        "Cross-validation is a technique for assessing how well a model generalizes to unseen data.",
        "Vector databases store high-dimensional vectors for efficient similarity search operations.",
        "BM25 is a probabilistic ranking function used in information retrieval systems.",
        "Semantic search understands query intent beyond exact keyword matching.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data."
    ]
    
    metadatas = [
        {"topic": "machine_learning", "category": "ai", "complexity": "beginner"},
        {"topic": "python_basics", "category": "programming", "complexity": "beginner"},
        {"topic": "fastapi", "category": "programming", "complexity": "intermediate"},
        {"topic": "deep_learning", "category": "ai", "complexity": "advanced"},
        {"topic": "pandas", "category": "programming", "complexity": "intermediate"},
        {"topic": "nlp", "category": "ai", "complexity": "intermediate"},
        {"topic": "feature_engineering", "category": "data_science", "complexity": "intermediate"},
        {"topic": "cross_validation", "category": "data_science", "complexity": "intermediate"},
        {"topic": "vector_databases", "category": "search", "complexity": "advanced"},
        {"topic": "bm25", "category": "search", "complexity": "advanced"},
        {"topic": "semantic_search", "category": "search", "complexity": "advanced"},
        {"topic": "data_science", "category": "data_science", "complexity": "beginner"}
    ]
    
    # Add documents to BM25s store
    bm25s_store.add_texts(documents, metadatas)
    print(f"Added {len(documents)} documents to BM25s store")
    
    # Create hybrid retriever
    hybrid_retriever = BM25sHybridRetriever(
        bm25s_store=bm25s_store,
        bm25s_weight=0.6,  # Favor BM25s for technical queries
        semantic_weight=0.4
    )
    
    print("Hybrid retriever initialized with 60% BM25s, 40% semantic weighting\n")
    
    return hybrid_retriever


def demonstrate_hybrid_search(retriever: BM25sHybridRetriever):
    """Demonstrate hybrid search capabilities."""
    
    print("=== Hybrid Search Demonstrations ===\n")
    
    test_queries = [
        {
            "query": "machine learning algorithms",
            "type": "Conceptual query (should favor semantic)",
            "expected": "AI and data science documents"
        },
        {
            "query": "Python FastAPI framework",
            "type": "Exact keyword query (should favor BM25s)",
            "expected": "Programming documents with exact matches"
        },
        {
            "query": "improve model performance",
            "type": "Intent-based query (hybrid approach)",
            "expected": "Feature engineering and validation documents"
        },
        {
            "query": "search and retrieval systems",
            "type": "Domain-specific query",
            "expected": "BM25, vector databases, and search documents"
        }
    ]
    
    for i, query_info in enumerate(test_queries, 1):
        query = query_info["query"]
        query_type = query_info["type"]
        expected = query_info["expected"]
        
        print(f"Query {i}: {query_type}")
        print(f"Search: '{query}'")
        print(f"Expected: {expected}")
        print("-" * 60)
        
        # Get hybrid results
        results = retriever.retrieve(query, k=4)
        
        bm25s_contribution = 0
        semantic_contribution = 0
        
        for rank, (doc, combined_score) in enumerate(results, 1):
            topic = doc.metadata.get('topic', 'unknown')
            category = doc.metadata.get('category', 'unknown')
            complexity = doc.metadata.get('complexity', 'unknown')
            
            print(f"  Rank {rank}: {topic} ({category}/{complexity})")
            print(f"    Combined Score: {combined_score:.3f}")
            print(f"    Content: {doc.page_content[:80]}...")
            
            # Estimate contribution (simplified)
            if combined_score > 0.5:
                bm25s_contribution += 1
            else:
                semantic_contribution += 1
            
            print()
        
        print(f"Search Strategy Analysis:")
        print(f"  BM25s-favored results: {bm25s_contribution}")
        print(f"  Semantic-favored results: {semantic_contribution}")
        print()
        print("=" * 80 + "\n")


def compare_search_strategies():
    """Compare different search strategies."""
    
    print("=== Search Strategy Comparison ===\n")
    
    comparison_data = [
        {
            "Strategy": "BM25s Only",
            "Strengths": [
                "Excellent exact keyword matching",
                "Fast and deterministic",
                "No embedding computation needed",
                "Works well with technical terms"
            ],
            "Weaknesses": [
                "Limited semantic understanding",
                "Vocabulary mismatch issues",
                "Poor with synonyms and paraphrases"
            ],
            "Best For": "Technical docs, exact term searches, FAQ systems"
        },
        {
            "Strategy": "Semantic Only",
            "Strengths": [
                "Understanding context and intent",
                "Handles synonyms and paraphrases",
                "Cross-lingual capabilities",
                "Conceptual similarity"
            ],
            "Weaknesses": [
                "Computationally expensive",
                "May miss exact keywords",
                "Less interpretable results",
                "Requires good embeddings"
            ],
            "Best For": "Conceptual queries, cross-lingual search, recommendation"
        },
        {
            "Strategy": "Hybrid (BM25s + Semantic)",
            "Strengths": [
                "Best of both worlds",
                "Robust across query types",
                "Tunable weights",
                "Complementary strengths"
            ],
            "Weaknesses": [
                "More complex implementation",
                "Requires tuning",
                "Higher computational cost",
                "Score normalization challenges"
            ],
            "Best For": "Production RAG systems, diverse query workloads"
        }
    ]
    
    for strategy in comparison_data:
        print(f"📊 {strategy['Strategy']}")
        print(f"   Best For: {strategy['Best For']}")
        print("   Strengths:")
        for strength in strategy['Strengths']:
            print(f"     ✓ {strength}")
        print("   Weaknesses:")
        for weakness in strategy['Weaknesses']:
            print(f"     ✗ {weakness}")
        print()


def production_recommendations():
    """Provide recommendations for production deployment."""
    
    print("=== Production Deployment Recommendations ===\n")
    
    print("🏗️ Architecture Recommendations:")
    print("1. Query Routing:")
    print("   • Analyze query patterns to route appropriately")
    print("   • Use keyword density for BM25s vs semantic routing")
    print("   • Implement fallback mechanisms")
    print()
    
    print("2. Performance Optimization:")
    print("   • Cache frequently accessed BM25s indices")
    print("   • Batch process semantic embeddings")
    print("   • Use async processing for hybrid searches")
    print("   • Implement result caching")
    print()
    
    print("3. Quality Assurance:")
    print("   • A/B test different weight combinations")
    print("   • Monitor retrieval quality metrics")
    print("   • Implement human feedback loops")
    print("   • Regular index maintenance")
    print()
    
    print("4. Scalability:")
    print("   • Distribute BM25s indices across nodes")
    print("   • Use approximate nearest neighbor for semantic search")
    print("   • Implement incremental index updates")
    print("   • Monitor memory and CPU usage")
    print()
    
    print("🔧 Configuration Best Practices:")
    print("• Start with 70% BM25s, 30% semantic for technical domains")
    print("• Adjust weights based on query analysis")
    print("• Use higher BM25s weights for exact match scenarios")
    print("• Increase semantic weights for conceptual queries")
    print()


async def main():
    """Main async function for hybrid search demonstration."""
    
    print("🔍 Hybrid Search with BM25s + Semantic Demo")
    print("=" * 50 + "\n")
    
    # Set up hybrid system
    hybrid_retriever = setup_hybrid_rag_system()
    
    # Demonstrate hybrid search
    demonstrate_hybrid_search(hybrid_retriever)
    
    # Compare strategies
    compare_search_strategies()
    
    # Production recommendations
    production_recommendations()
    
    print("🎉 Hybrid search demo completed!")
    print("\nNext Steps:")
    print("1. Integrate real semantic embeddings (OpenAI, Sentence-BERT)")
    print("2. Implement proper score normalization")
    print("3. Add query analysis for automatic weight adjustment")
    print("4. Set up evaluation metrics and benchmarking")


if __name__ == "__main__":
    asyncio.run(main())
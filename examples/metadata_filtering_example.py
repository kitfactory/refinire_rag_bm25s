"""Metadata filtering example using BM25s-j 0.2.0+ features."""

from typing import List, Dict, Any

from refinire_rag_bm25s_j import BM25sStore
from refinire_rag_bm25s_j.models import BM25sConfig


def create_sample_documents_with_metadata() -> tuple[List[str], List[Dict[str, Any]]]:
    """Create sample documents with rich metadata for filtering demonstrations."""
    
    documents = [
        # Technical Documents
        "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including object-oriented, procedural, and functional programming.",
        
        "Machine learning algorithms can automatically learn patterns from data without being explicitly programmed. Popular algorithms include linear regression, decision trees, and neural networks.",
        
        "RESTful APIs provide a standardized way for different software systems to communicate over HTTP. They use standard HTTP methods like GET, POST, PUT, and DELETE.",
        
        "Docker containers provide a lightweight alternative to virtual machines by packaging applications with their dependencies into portable, isolated environments.",
        
        # Business Documents
        "The quarterly financial report shows a 15% increase in revenue compared to the previous quarter. Operating expenses remained stable while profit margins improved significantly.",
        
        "Our marketing strategy for 2024 focuses on digital transformation and customer engagement through social media platforms and personalized content delivery.",
        
        "The new employee onboarding process includes orientation sessions, mentorship programs, and skill assessment tests to ensure smooth integration into company culture.",
        
        # Academic Documents
        "The research paper investigates the effects of climate change on marine ecosystems using statistical analysis of temperature and biodiversity data collected over 20 years.",
        
        "Advanced calculus concepts include multivariable functions, partial derivatives, and vector calculus applications in physics and engineering problems.",
        
        "The psychology study examines cognitive behavioral therapy effectiveness in treating anxiety disorders through randomized controlled trials with 500 participants.",
        
        # Legal Documents
        "The software license agreement grants users the right to use the application for commercial purposes while maintaining intellectual property protections for the developer.",
        
        "Data privacy regulations require companies to implement appropriate technical and organizational measures to protect personal information from unauthorized access and processing."
    ]
    
    metadatas = [
        # Technical Documents
        {"category": "technology", "language": "python", "difficulty": "beginner", "type": "tutorial", "author": "tech_team", "year": 2024, "rating": 4.5},
        {"category": "technology", "language": "python", "difficulty": "intermediate", "type": "guide", "author": "data_team", "year": 2024, "rating": 4.8},
        {"category": "technology", "language": "api", "difficulty": "intermediate", "type": "documentation", "author": "backend_team", "year": 2023, "rating": 4.2},
        {"category": "technology", "language": "devops", "difficulty": "advanced", "type": "tutorial", "author": "devops_team", "year": 2024, "rating": 4.7},
        
        # Business Documents
        {"category": "business", "department": "finance", "difficulty": "intermediate", "type": "report", "author": "finance_team", "year": 2024, "rating": 4.0},
        {"category": "business", "department": "marketing", "difficulty": "beginner", "type": "strategy", "author": "marketing_team", "year": 2024, "rating": 3.9},
        {"category": "business", "department": "hr", "difficulty": "beginner", "type": "process", "author": "hr_team", "year": 2023, "rating": 4.1},
        
        # Academic Documents
        {"category": "academic", "field": "environmental_science", "difficulty": "advanced", "type": "research", "author": "research_team", "year": 2023, "rating": 4.9},
        {"category": "academic", "field": "mathematics", "difficulty": "advanced", "type": "textbook", "author": "math_dept", "year": 2022, "rating": 4.6},
        {"category": "academic", "field": "psychology", "difficulty": "intermediate", "type": "study", "author": "psychology_dept", "year": 2024, "rating": 4.4},
        
        # Legal Documents
        {"category": "legal", "type": "license", "difficulty": "advanced", "jurisdiction": "international", "author": "legal_team", "year": 2024, "rating": 4.3},
        {"category": "legal", "type": "regulation", "difficulty": "advanced", "jurisdiction": "eu", "author": "compliance_team", "year": 2023, "rating": 4.8}
    ]
    
    return documents, metadatas


def demonstrate_basic_filtering():
    """Demonstrate basic metadata filtering capabilities."""
    
    print("=== Basic Metadata Filtering Demo ===\\n")
    
    # Setup
    config = BM25sConfig(
        k1=1.2,
        b=0.75,
        epsilon=0.25,
        index_path="./data/metadata_filtering_index.pkl"
    )
    
    vector_store = BM25sStore(config=config)
    documents, metadatas = create_sample_documents_with_metadata()
    
    # Add documents with metadata
    print("📚 Adding documents with metadata...")
    doc_ids = vector_store.add_texts(documents, metadatas)
    print(f"Added {len(doc_ids)} documents\\n")
    
    # Test 1: Category filtering
    print("Test 1: Technology documents only")
    print("Filter: {'category': 'technology'}")
    print("-" * 50)
    
    tech_results = vector_store.similarity_search(
        query="programming languages",
        k=5,
        filter={"category": "technology"}
    )
    
    for i, doc in enumerate(tech_results, 1):
        category = doc.metadata.get('category', 'unknown')
        language = doc.metadata.get('language', 'unknown')
        difficulty = doc.metadata.get('difficulty', 'unknown')
        print(f"{i}. {category}/{language}/{difficulty}")
        print(f"   {doc.page_content[:80]}...")
        print(f"   Score: {doc.metadata.get('score', 'N/A'):.3f}")
        print()
    
    # Test 2: Multiple criteria filtering
    print("Test 2: Advanced technology documents from 2024")
    print("Filter: {'category': 'technology', 'difficulty': 'advanced', 'year': 2024}")
    print("-" * 70)
    
    advanced_tech_results = vector_store.similarity_search(
        query="containers deployment",
        k=3,
        filter={"category": "technology", "difficulty": "advanced", "year": 2024}
    )
    
    for i, doc in enumerate(advanced_tech_results, 1):
        print(f"{i}. {doc.metadata.get('language', 'unknown')} - {doc.metadata.get('type', 'unknown')}")
        print(f"   {doc.page_content[:80]}...")
        print(f"   Author: {doc.metadata.get('author', 'unknown')}")
        print()
    
    # Test 3: List-based filtering
    print("Test 3: Business and Academic documents")
    print("Filter: {'category': ['business', 'academic']}")
    print("-" * 50)
    
    business_academic_results = vector_store.similarity_search(
        query="research analysis data",
        k=4,
        filter={"category": ["business", "academic"]}
    )
    
    for i, doc in enumerate(business_academic_results, 1):
        category = doc.metadata.get('category', 'unknown')
        field_or_dept = doc.metadata.get('field', doc.metadata.get('department', 'unknown'))
        print(f"{i}. {category}/{field_or_dept}")
        print(f"   {doc.page_content[:80]}...")
        print()


def demonstrate_advanced_filtering():
    """Demonstrate advanced filtering with operators."""
    
    print("=== Advanced Metadata Filtering Demo ===\\n")
    
    config = BM25sConfig(index_path="./data/advanced_filtering_index.pkl")
    vector_store = BM25sStore(config=config)
    documents, metadatas = create_sample_documents_with_metadata()
    vector_store.add_texts(documents, metadatas)
    
    # Test 1: Comparison operators
    print("Test 1: High-rated documents (rating >= 4.5)")
    print("Filter: {'rating': {'$gte': 4.5}}")
    print("-" * 50)
    
    high_rated_results = vector_store.similarity_search_with_score(
        query="programming machine learning",
        k=5,
        filter={"rating": {"$gte": 4.5}}
    )
    
    for i, (doc, score) in enumerate(high_rated_results, 1):
        rating = doc.metadata.get('rating', 0)
        category = doc.metadata.get('category', 'unknown')
        print(f"{i}. {category} (Rating: {rating})")
        print(f"   BM25 Score: {score:.3f}")
        print(f"   {doc.page_content[:80]}...")
        print()
    
    # Test 2: Range filtering
    print("Test 2: Recent documents (2023-2024)")
    print("Filter: {'year': {'$gte': 2023}}")
    print("-" * 40)
    
    recent_results = vector_store.similarity_search(
        query="new developments trends",
        k=4,
        filter={"year": {"$gte": 2023}}
    )
    
    for i, doc in enumerate(recent_results, 1):
        year = doc.metadata.get('year', 'unknown')
        author = doc.metadata.get('author', 'unknown')
        print(f"{i}. {year} - {author}")
        print(f"   {doc.page_content[:70]}...")
        print()
    
    # Test 3: Exclusion filtering
    print("Test 3: Non-beginner documents")
    print("Filter: {'difficulty': {'$ne': 'beginner'}}")
    print("-" * 45)
    
    non_beginner_results = vector_store.similarity_search(
        query="complex advanced analysis",
        k=4,
        filter={"difficulty": {"$ne": "beginner"}}
    )
    
    for i, doc in enumerate(non_beginner_results, 1):
        difficulty = doc.metadata.get('difficulty', 'unknown')
        category = doc.metadata.get('category', 'unknown')
        print(f"{i}. {category}/{difficulty}")
        print(f"   {doc.page_content[:70]}...")
        print()


def demonstrate_real_world_scenarios():
    """Demonstrate real-world filtering scenarios."""
    
    print("=== Real-World Filtering Scenarios ===\\n")
    
    config = BM25sConfig(index_path="./data/realworld_filtering_index.pkl")
    vector_store = BM25sStore(config=config)
    documents, metadatas = create_sample_documents_with_metadata()
    vector_store.add_texts(documents, metadatas)
    
    scenarios = [
        {
            "name": "Developer looking for Python tutorials",
            "query": "python programming tutorial",
            "filter": {"category": "technology", "language": "python", "type": "tutorial"},
            "description": "Filter for technology documents specifically about Python tutorials"
        },
        {
            "name": "Manager seeking recent business reports",
            "query": "quarterly performance analysis",
            "filter": {"category": "business", "year": 2024, "rating": {"$gte": 4.0}},
            "description": "Filter for high-quality business documents from current year"
        },
        {
            "name": "Researcher looking for environmental studies",
            "query": "climate environmental research",
            "filter": {"category": "academic", "field": "environmental_science", "difficulty": "advanced"},
            "description": "Filter for advanced academic research in environmental science"
        },
        {
            "name": "Compliance officer checking regulations",
            "query": "data privacy compliance",
            "filter": {"category": "legal", "type": "regulation"},
            "description": "Filter for legal documents specifically about regulations"
        },
        {
            "name": "Team lead finding intermediate resources",
            "query": "team development processes",
            "filter": {"difficulty": ["intermediate", "advanced"], "year": {"$gte": 2023}},
            "description": "Filter for recent intermediate to advanced level content"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Query: '{scenario['query']}'")
        print(f"Filter: {scenario['filter']}")
        print("-" * 60)
        
        results = vector_store.similarity_search_with_score(
            query=scenario["query"],
            k=2,
            filter=scenario["filter"]
        )
        
        if results:
            for j, (doc, score) in enumerate(results, 1):
                print(f"  Result {j} (Score: {score:.3f}):")
                print(f"    Category: {doc.metadata.get('category', 'unknown')}")
                print(f"    Type: {doc.metadata.get('type', 'unknown')}")
                print(f"    Difficulty: {doc.metadata.get('difficulty', 'unknown')}")
                print(f"    Content: {doc.page_content[:60]}...")
                print()
        else:
            print("  No matching documents found.")
            print()
        
        print("=" * 60 + "\\n")


def performance_comparison():
    """Compare performance with and without metadata filtering."""
    
    print("=== Performance Comparison ===\\n")
    
    import time
    
    config = BM25sConfig(index_path="./data/performance_test_index.pkl")
    vector_store = BM25sStore(config=config)
    documents, metadatas = create_sample_documents_with_metadata()
    vector_store.add_texts(documents, metadatas)
    
    query = "programming technology development"
    
    # Test without filtering
    print("🔍 Search without filtering:")
    start_time = time.time()
    all_results = vector_store.similarity_search(query, k=5)
    no_filter_time = time.time() - start_time
    print(f"  Results: {len(all_results)}")
    print(f"  Time: {no_filter_time:.4f} seconds")
    print()
    
    # Test with filtering
    print("🔍 Search with category filtering:")
    start_time = time.time()
    filtered_results = vector_store.similarity_search(
        query, k=5, filter={"category": "technology"}
    )
    filter_time = time.time() - start_time
    print(f"  Results: {len(filtered_results)}")
    print(f"  Time: {filter_time:.4f} seconds")
    print()
    
    # Test with complex filtering
    print("🔍 Search with complex filtering:")
    start_time = time.time()
    complex_filtered_results = vector_store.similarity_search(
        query, k=5, filter={
            "category": "technology", 
            "difficulty": ["intermediate", "advanced"],
            "rating": {"$gte": 4.0}
        }
    )
    complex_filter_time = time.time() - start_time
    print(f"  Results: {len(complex_filtered_results)}")
    print(f"  Time: {complex_filter_time:.4f} seconds")
    print()
    
    print("📊 Performance Summary:")
    print(f"  No filtering: {no_filter_time:.4f}s")
    print(f"  Basic filtering: {filter_time:.4f}s")
    print(f"  Complex filtering: {complex_filter_time:.4f}s")
    
    if hasattr(vector_store.search_service, '_supports_native_filtering'):
        native_support = vector_store.search_service._supports_native_filtering()
        print(f"  Native filtering support: {'Yes' if native_support else 'No (using fallback)'}")


def main():
    """Main function to run all metadata filtering demonstrations."""
    
    print("🔍 BM25s-j 0.2.0+ Metadata Filtering Demonstrations")
    print("=" * 60 + "\\n")
    
    # Run demonstrations
    demonstrate_basic_filtering()
    print("\\n" + "=" * 60 + "\\n")
    
    demonstrate_advanced_filtering()
    print("\\n" + "=" * 60 + "\\n")
    
    demonstrate_real_world_scenarios()
    print("\\n" + "=" * 60 + "\\n")
    
    performance_comparison()
    
    print("\\n🎉 Metadata filtering demonstrations completed!")
    print("\\nKey Features Demonstrated:")
    print("✅ Basic equality filtering")
    print("✅ Multiple criteria filtering")
    print("✅ List-based filtering (OR conditions)")
    print("✅ Comparison operators ($gte, $ne, etc.)")
    print("✅ Real-world use case scenarios")
    print("✅ Performance impact analysis")


if __name__ == "__main__":
    main()
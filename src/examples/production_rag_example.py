"""Production-ready RAG example using BM25s with refinire-rag framework."""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from refinire import Refinire
    from refinire.rag import RAGPipeline, RAGConfig
    from refinire.document_loader import DocumentLoader
    from refinire.text_splitter import RecursiveCharacterTextSplitter
    from refinire.llm import LLMProvider
    from langchain_core.documents import Document
except ImportError:
    print("This example requires refinire-rag. Install it with: uv add refinire-rag")
    exit(1)

from refinire_rag_bm25s_j import BM25sStore
from refinire_rag_bm25s_j.models import BM25sConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGMetrics:
    """Metrics for RAG system performance."""
    retrieval_time: float
    generation_time: float
    total_time: float
    documents_retrieved: int
    context_length: int
    query_length: int


class ProductionBM25sRAG:
    """Production-ready RAG system using BM25s VectorStore."""
    
    def __init__(
        self,
        bm25s_config: Optional[BM25sConfig] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_context_length: int = 4000
    ):
        """Initialize production RAG system."""
        self.bm25s_config = bm25s_config or BM25sConfig(
            k1=1.2,
            b=0.75,
            epsilon=0.25,
            index_path="./data/production_bm25s_index.pkl"
        )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_length = max_context_length
        
        # Initialize components
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\\n\\n", "\\n", ". ", " "]
        )
        
        # Metrics tracking
        self.metrics_history: List[RAGMetrics] = []
        
        logger.info("ProductionBM25sRAG initialized")
    
    async def initialize(self):
        """Initialize the RAG system components."""
        logger.info("Initializing RAG system...")
        
        # Initialize vector store
        self.vector_store = BM25sStore(config=self.bm25s_config)
        
        # Try to load existing index
        try:
            if Path(self.bm25s_config.index_path).exists():
                logger.info("Loading existing BM25s index")
                # Index is automatically loaded during initialization
            else:
                logger.info("No existing index found, will create new one")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
        
        logger.info("RAG system initialized successfully")
    
    async def ingest_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> int:
        """Ingest documents into the RAG system with batching."""
        logger.info(f"Starting document ingestion: {len(documents)} documents")
        
        if not self.vector_store:
            await self.initialize()
        
        # Split documents into chunks
        all_chunks = []
        all_chunk_metadata = []
        
        for i, doc_text in enumerate(documents):
            # Split document
            chunks = self.text_splitter.split_text(doc_text)
            
            # Create metadata for each chunk
            base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            for j, chunk in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    "doc_id": i,
                    "chunk_id": j,
                    "chunk_count": len(chunks),
                    "chunk_size": len(chunk)
                }
                
                all_chunks.append(chunk)
                all_chunk_metadata.append(chunk_metadata)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Ingest in batches
        total_ingested = 0
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_metadata = all_chunk_metadata[i:i + batch_size]
            
            start_time = time.time()
            doc_ids = self.vector_store.add_texts(batch_chunks, batch_metadata)
            ingestion_time = time.time() - start_time
            
            total_ingested += len(doc_ids)
            logger.info(
                f"Ingested batch {i//batch_size + 1}: "
                f"{len(doc_ids)} chunks in {ingestion_time:.2f}s"
            )
        
        logger.info(f"Document ingestion completed: {total_ingested} chunks total")
        return total_ingested
    
    async def retrieve_context(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Document], float]:
        """Retrieve relevant context for a query."""
        start_time = time.time()
        
        if not self.vector_store:
            await self.initialize()
        
        # Retrieve documents
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Apply metadata filtering if specified
        if filter_metadata:
            filtered_results = []
            for doc, score in results:
                if all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                ):
                    filtered_results.append((doc, score))
            results = filtered_results
        
        retrieval_time = time.time() - start_time
        
        # Extract documents
        documents = [doc for doc, _ in results]
        
        logger.info(
            f"Retrieved {len(documents)} documents in {retrieval_time:.3f}s "
            f"for query: '{query[:50]}...'"
        )
        
        return documents, retrieval_time
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # Add source information
            source_info = (
                f"Source {i+1} "
                f"(Doc {doc.metadata.get('doc_id', 'unknown')}, "
                f"Chunk {doc.metadata.get('chunk_id', 'unknown')}):\\n"
            )
            
            content = doc.page_content
            
            # Check if adding this document would exceed max context length
            estimated_length = current_length + len(source_info) + len(content) + 10
            
            if estimated_length > self.max_context_length and context_parts:
                logger.info(f"Context length limit reached, using {len(context_parts)} documents")
                break
            
            context_parts.append(source_info + content)
            current_length = estimated_length
        
        return "\\n\\n".join(context_parts)
    
    async def generate_response(
        self,
        query: str,
        context: str,
        temperature: float = 0.1
    ) -> tuple[str, float]:
        """Generate response using LLM (simulated for this example)."""
        start_time = time.time()
        
        # In a real implementation, this would call an actual LLM
        # For this example, we'll simulate response generation
        
        prompt = f"""Based on the following context, answer the question accurately and concisely.
        
Context:
{context}

Question: {query}

Answer:"""
        
        # Simulate LLM processing time
        await asyncio.sleep(0.5)  # Simulate API call delay
        
        # Simulated response
        response = f"""Based on the provided context, I can provide the following information about "{query}":

[This would be the actual LLM response in a production system]

The context contains {len(context.split())} words from multiple relevant sources that help answer your question about {query.split()[:3]}."""
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generated response in {generation_time:.3f}s")
        
        return response, generation_time
    
    async def query(
        self,
        question: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Process a complete RAG query."""
        total_start_time = time.time()
        
        logger.info(f"Processing RAG query: '{question}'")
        
        # Retrieve context
        documents, retrieval_time = await self.retrieve_context(
            question, k=k, filter_metadata=filter_metadata
        )
        
        if not documents:
            logger.warning("No relevant documents found")
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "metrics": RAGMetrics(
                    retrieval_time=retrieval_time,
                    generation_time=0.0,
                    total_time=time.time() - total_start_time,
                    documents_retrieved=0,
                    context_length=0,
                    query_length=len(question)
                )
            }
        
        # Build context
        context = self._build_context(documents)
        
        # Generate response
        response, generation_time = await self.generate_response(
            question, context, temperature
        )
        
        total_time = time.time() - total_start_time
        
        # Create metrics
        metrics = RAGMetrics(
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            documents_retrieved=len(documents),
            context_length=len(context),
            query_length=len(question)
        )
        
        self.metrics_history.append(metrics)
        
        # Prepare sources information
        sources = [
            {
                "doc_id": doc.metadata.get("doc_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "category": doc.metadata.get("category"),
                "content_preview": doc.page_content[:100] + "..."
            }
            for doc in documents
        ]
        
        logger.info(f"RAG query completed in {total_time:.3f}s")
        
        return {
            "answer": response,
            "sources": sources,
            "metrics": metrics,
            "context_used": len(context) < self.max_context_length
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics_history:
            return {"message": "No queries processed yet"}
        
        metrics = self.metrics_history
        
        return {
            "total_queries": len(metrics),
            "avg_retrieval_time": sum(m.retrieval_time for m in metrics) / len(metrics),
            "avg_generation_time": sum(m.generation_time for m in metrics) / len(metrics),
            "avg_total_time": sum(m.total_time for m in metrics) / len(metrics),
            "avg_documents_retrieved": sum(m.documents_retrieved for m in metrics) / len(metrics),
            "avg_context_length": sum(m.context_length for m in metrics) / len(metrics),
            "max_total_time": max(m.total_time for m in metrics),
            "min_total_time": min(m.total_time for m in metrics)
        }


def create_sample_knowledge_base() -> tuple[List[str], List[Dict[str, Any]]]:
    """Create a comprehensive sample knowledge base."""
    
    documents = [
        # Software Engineering
        """
        Software engineering is a systematic approach to the design, development, and maintenance of software systems. 
        It involves applying engineering principles to software development, including requirements analysis, system design, 
        implementation, testing, and maintenance. Key methodologies include Agile, Waterfall, and DevOps practices. 
        Modern software engineering emphasizes continuous integration, automated testing, and collaborative development 
        through version control systems like Git.
        """,
        
        # Machine Learning
        """
        Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience 
        without being explicitly programmed. It includes supervised learning (with labeled data), unsupervised learning 
        (finding patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment). 
        Common algorithms include linear regression, decision trees, neural networks, and support vector machines. 
        Feature engineering, model validation, and hyperparameter tuning are crucial aspects of the ML workflow.
        """,
        
        # Data Science
        """
        Data science is an interdisciplinary field that combines statistics, computer science, and domain expertise 
        to extract insights from data. The data science process typically involves data collection, cleaning, 
        exploratory data analysis, modeling, and interpretation. Key tools include Python, R, SQL, and visualization 
        libraries like Matplotlib and Plotly. Data scientists work with structured and unstructured data to solve 
        business problems and inform decision-making processes.
        """,
        
        # Web Development
        """
        Web development encompasses the creation of websites and web applications. Frontend development focuses on 
        user interfaces using HTML, CSS, and JavaScript frameworks like React, Vue, or Angular. Backend development 
        involves server-side logic, databases, and APIs using languages like Python, Java, or Node.js. Modern web 
        development follows responsive design principles, emphasizes performance optimization, and implements security 
        best practices including HTTPS, authentication, and input validation.
        """,
        
        # Database Systems
        """
        Database systems are organized collections of data that support efficient storage, retrieval, and management 
        of information. Relational databases use SQL and follow ACID properties, while NoSQL databases offer flexibility 
        for unstructured data. Key concepts include normalization, indexing, query optimization, and transaction management. 
        Modern database technologies include distributed systems, cloud databases, and specialized databases for 
        time-series data, graphs, and document storage.
        """,
        
        # DevOps and Cloud
        """
        DevOps is a cultural and technical movement that emphasizes collaboration between development and operations teams. 
        It includes practices like continuous integration/continuous deployment (CI/CD), infrastructure as code, 
        monitoring, and automation. Cloud computing provides scalable infrastructure through services like AWS, Azure, 
        and Google Cloud. Container technologies like Docker and orchestration platforms like Kubernetes enable 
        efficient application deployment and management.
        """,
        
        # Cybersecurity
        """
        Cybersecurity involves protecting digital systems, networks, and data from threats and attacks. Key areas include 
        network security, application security, endpoint protection, and incident response. Common threats include malware, 
        phishing, SQL injection, and social engineering attacks. Security measures include encryption, access controls, 
        security audits, and employee training. Compliance frameworks like GDPR, HIPAA, and SOX provide guidelines 
        for data protection and privacy.
        """,
        
        # API Development
        """
        Application Programming Interfaces (APIs) enable communication between different software systems. REST APIs 
        use HTTP methods and status codes for stateless communication, while GraphQL provides flexible data querying. 
        API design principles include consistent naming conventions, proper error handling, versioning strategies, 
        and comprehensive documentation. Security considerations include authentication (OAuth, JWT), rate limiting, 
        and input validation. API testing and monitoring are essential for maintaining reliability and performance.
        """
    ]
    
    metadatas = [
        {"category": "software_engineering", "topic": "methodology", "difficulty": "intermediate", "domain": "technology"},
        {"category": "ai_ml", "topic": "machine_learning", "difficulty": "intermediate", "domain": "data_science"},
        {"category": "data_science", "topic": "analytics", "difficulty": "beginner", "domain": "data_science"},
        {"category": "web_development", "topic": "full_stack", "difficulty": "intermediate", "domain": "technology"},
        {"category": "databases", "topic": "data_management", "difficulty": "intermediate", "domain": "data_engineering"},
        {"category": "devops", "topic": "infrastructure", "difficulty": "advanced", "domain": "operations"},
        {"category": "security", "topic": "cybersecurity", "difficulty": "advanced", "domain": "security"},
        {"category": "api", "topic": "integration", "difficulty": "intermediate", "domain": "technology"}
    ]
    
    return documents, metadatas


async def demonstrate_production_rag():
    """Demonstrate production RAG system capabilities."""
    
    print("🏭 Production RAG System with BM25s Demo")
    print("=" * 50 + "\n")
    
    # Initialize RAG system
    rag_system = ProductionBM25sRAG(
        chunk_size=800,
        chunk_overlap=100,
        max_context_length=3000
    )
    
    await rag_system.initialize()
    
    # Ingest knowledge base
    documents, metadatas = create_sample_knowledge_base()
    
    print("📚 Ingesting knowledge base...")
    chunks_ingested = await rag_system.ingest_documents(documents, metadatas, batch_size=50)
    print(f"✅ Ingested {chunks_ingested} chunks\\n")
    
    # Test queries
    test_queries = [
        {
            "question": "What are the key principles of software engineering?",
            "filter": None
        },
        {
            "question": "How does machine learning work?",
            "filter": {"domain": "data_science"}
        },
        {
            "question": "What security measures should I implement for web applications?",
            "filter": {"category": "security"}
        },
        {
            "question": "Explain the difference between SQL and NoSQL databases",
            "filter": {"topic": "data_management"}
        },
        {
            "question": "What are the benefits of DevOps practices?",
            "filter": None
        }
    ]
    
    print("🔍 Processing queries...")
    print("=" * 60 + "\\n")
    
    for i, query_info in enumerate(test_queries, 1):
        question = query_info["question"]
        metadata_filter = query_info["filter"]
        
        print(f"Query {i}: {question}")
        if metadata_filter:
            print(f"Filter: {metadata_filter}")
        print("-" * 50)
        
        # Process query
        result = await rag_system.query(
            question=question,
            k=4,
            filter_metadata=metadata_filter
        )
        
        # Display results
        print("📋 Answer:")
        print(f"  {result['answer'][:200]}...")
        print()
        
        print("📊 Sources Used:")
        for j, source in enumerate(result['sources'], 1):
            category = source.get('category', 'unknown')
            print(f"  {j}. {category} - {source['content_preview']}")
        print()
        
        metrics = result['metrics']
        print("⏱️ Performance:")
        print(f"  Retrieval: {metrics.retrieval_time:.3f}s")
        print(f"  Generation: {metrics.generation_time:.3f}s")
        print(f"  Total: {metrics.total_time:.3f}s")
        print(f"  Documents: {metrics.documents_retrieved}")
        print(f"  Context length: {metrics.context_length} chars")
        print()
        print("=" * 60 + "\\n")
    
    # Show performance statistics
    stats = rag_system.get_performance_stats()
    print("📈 Overall Performance Statistics:")
    print(f"  Total queries processed: {stats['total_queries']}")
    print(f"  Average retrieval time: {stats['avg_retrieval_time']:.3f}s")
    print(f"  Average generation time: {stats['avg_generation_time']:.3f}s")
    print(f"  Average total time: {stats['avg_total_time']:.3f}s")
    print(f"  Average documents per query: {stats['avg_documents_retrieved']:.1f}")
    print(f"  Average context length: {stats['avg_context_length']:.0f} chars")
    print()


def print_production_guidelines():
    """Print guidelines for production deployment."""
    
    print("🚀 Production Deployment Guidelines")
    print("=" * 50 + "\\n")
    
    print("🔧 Configuration:")
    print("• Set appropriate chunk sizes based on your domain (500-1500 tokens)")
    print("• Configure BM25s parameters (k1=1.2, b=0.75 for general use)")
    print("• Set reasonable context limits to avoid LLM token limits")
    print("• Implement proper error handling and fallbacks")
    print()
    
    print("📊 Monitoring:")
    print("• Track retrieval and generation latencies")
    print("• Monitor relevance scores and user feedback")
    print("• Set up alerts for system health")
    print("• Log queries for analysis and improvement")
    print()
    
    print("🔒 Security:")
    print("• Implement input validation and sanitization")
    print("• Use authentication and authorization")
    print("• Protect sensitive information in documents")
    print("• Regular security audits and updates")
    print()
    
    print("📈 Scalability:")
    print("• Use async processing for better concurrency")
    print("• Implement caching for frequent queries")
    print("• Consider distributed deployment for high load")
    print("• Plan for index growth and maintenance")
    print()


async def main():
    """Main function for production RAG demonstration."""
    
    # Run the demonstration
    await demonstrate_production_rag()
    
    # Show production guidelines
    print_production_guidelines()
    
    print("🎉 Production RAG demo completed!")
    print("\\nNext steps for production deployment:")
    print("1. Integrate with your preferred LLM provider")
    print("2. Set up proper logging and monitoring")
    print("3. Implement user feedback collection")
    print("4. Configure deployment infrastructure")
    print("5. Set up CI/CD pipelines for updates")


if __name__ == "__main__":
    asyncio.run(main())
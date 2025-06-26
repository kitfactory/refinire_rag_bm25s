"""Tests for BM25s VectorStore."""

from unittest.mock import Mock, patch

import pytest

from refinire_rag_bm25s_j.models import BM25sConfig, BM25sDocument
from refinire_rag_bm25s_j.vector_store import BM25sVectorStore


class TestBM25sVectorStore:
    """Test BM25sVectorStore class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BM25sConfig()
        self.vector_store = BM25sVectorStore(config=self.config)
    
    @patch('refinire_rag_bm25s_j.vector_store.BM25sIndexService')
    def test_initialization(self, mock_index_service):
        """Test vector store initialization."""
        config = BM25sConfig(index_path="/tmp/test.pkl")
        store = BM25sVectorStore(config=config)
        
        assert store.config == config
        mock_index_service.assert_called_once_with(config)
    
    def test_add_texts(self):
        """Test adding texts to vector store."""
        texts = ["First document", "Second document"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        
        # Mock the services
        self.vector_store.index_service.get_documents = Mock(return_value=[])
        self.vector_store.index_service.create_index = Mock()
        
        ids = self.vector_store.add_texts(texts, metadatas)
        
        assert len(ids) == 2
        assert ids[0].startswith("doc_")
        self.vector_store.index_service.create_index.assert_called_once()
    
    def test_add_texts_with_custom_ids(self):
        """Test adding texts with custom IDs."""
        texts = ["First document"]
        ids = ["custom-id-1"]
        
        self.vector_store.index_service.get_documents = Mock(return_value=[])
        self.vector_store.index_service.create_index = Mock()
        
        result_ids = self.vector_store.add_texts(texts, ids=ids)
        
        assert result_ids == ["custom-id-1"]
    
    def test_add_texts_empty_list(self):
        """Test adding empty texts list."""
        result = self.vector_store.add_texts([])
        assert result == []
    
    @patch('refinire_rag_bm25s_j.vector_store.Document')
    def test_add_documents(self, mock_document):
        """Test adding documents to vector store."""
        # Mock Document objects
        doc1 = Mock()
        doc1.page_content = "Document content 1"
        doc1.metadata = {"source": "test1"}
        
        doc2 = Mock()
        doc2.page_content = "Document content 2"
        doc2.metadata = {"source": "test2"}
        
        documents = [doc1, doc2]
        
        self.vector_store.index_service.get_documents = Mock(return_value=[])
        self.vector_store.index_service.create_index = Mock()
        
        ids = self.vector_store.add_documents(documents)
        
        assert len(ids) == 2
        self.vector_store.index_service.create_index.assert_called_once()
    
    @patch('refinire_rag_bm25s_j.vector_store.Document')
    def test_similarity_search(self, mock_document_class):
        """Test similarity search."""
        # Mock search results
        doc = BM25sDocument(id="test-1", content="Test content", metadata={"source": "test"})
        search_result = SearchResult(document=doc, score=0.9, rank=1)
        
        self.vector_store.search_service.search = Mock(return_value=[search_result])
        
        # Mock Document constructor
        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        
        results = self.vector_store.similarity_search("test query", k=1)
        
        assert len(results) == 1
        self.vector_store.search_service.search.assert_called_once_with("test query", top_k=1)
        mock_document_class.assert_called_once()
    
    @patch('refinire_rag_bm25s_j.vector_store.Document')
    def test_similarity_search_with_score(self, mock_document_class):
        """Test similarity search with scores."""
        doc = BM25sDocument(id="test-1", content="Test content")
        search_result = SearchResult(document=doc, score=0.85, rank=1)
        
        self.vector_store.search_service.search = Mock(return_value=[search_result])
        
        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        
        results = self.vector_store.similarity_search_with_score("test query", k=1)
        
        assert len(results) == 1
        doc_result, score = results[0]
        assert score == 0.85
        assert doc_result == mock_doc
    
    def test_max_marginal_relevance_search(self):
        """Test MMR search (fallback to similarity search)."""
        self.vector_store.similarity_search = Mock(return_value=["mock_result"])
        
        results = self.vector_store.max_marginal_relevance_search("test query")
        
        assert results == ["mock_result"]
        self.vector_store.similarity_search.assert_called_once_with("test query", 4)
    
    @patch('refinire_rag_bm25s_j.vector_store.BM25sVectorStore')
    def test_from_texts(self, mock_vector_store_class):
        """Test creating vector store from texts."""
        mock_store = Mock()
        mock_vector_store_class.return_value = mock_store
        
        texts = ["Text 1", "Text 2"]
        config = BM25sConfig()
        
        result = BM25sVectorStore.from_texts(texts, config=config)
        
        mock_vector_store_class.assert_called_once_with(config=config)
        mock_store.add_texts.assert_called_once_with(texts, None)
    
    @patch('refinire_rag_bm25s_j.vector_store.BM25sVectorStore')
    def test_from_documents(self, mock_vector_store_class):
        """Test creating vector store from documents."""
        mock_store = Mock()
        mock_vector_store_class.return_value = mock_store
        
        documents = [Mock(), Mock()]
        config = BM25sConfig()
        
        result = BM25sVectorStore.from_documents(documents, config=config)
        
        mock_vector_store_class.assert_called_once_with(config=config)
        mock_store.add_documents.assert_called_once_with(documents)
    
    def test_delete_documents(self):
        """Test deleting documents by IDs."""
        existing_docs = [
            BM25sDocument(id="doc1", content="Content 1"),
            BM25sDocument(id="doc2", content="Content 2"),
            BM25sDocument(id="doc3", content="Content 3")
        ]
        
        self.vector_store.index_service.get_documents = Mock(return_value=existing_docs)
        self.vector_store.index_service.create_index = Mock()
        
        result = self.vector_store.delete(["doc2"])
        
        assert result is True
        self.vector_store.index_service.create_index.assert_called_once()
        
        # Check that doc2 was filtered out
        call_args = self.vector_store.index_service.create_index.call_args[0][0]
        remaining_ids = [doc.id for doc in call_args]
        assert "doc2" not in remaining_ids
        assert "doc1" in remaining_ids
        assert "doc3" in remaining_ids
    
    def test_delete_nonexistent_documents(self):
        """Test deleting documents that don't exist."""
        existing_docs = [
            BM25sDocument(id="doc1", content="Content 1")
        ]
        
        self.vector_store.index_service.get_documents = Mock(return_value=existing_docs)
        
        result = self.vector_store.delete(["nonexistent"])
        
        assert result is False
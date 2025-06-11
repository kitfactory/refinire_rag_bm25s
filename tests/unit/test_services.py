"""Tests for service layer."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from refinire_rag_bm25s_j.models import BM25sConfig, BM25sDocument
from refinire_rag_bm25s_j.services import BM25sIndexService, BM25sSearchService


class TestBM25sIndexService:
    """Test BM25sIndexService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BM25sConfig(k1=1.2, b=0.75)
        self.service = BM25sIndexService(self.config)
        self.test_documents = [
            BM25sDocument(id="doc1", content="This is the first document"),
            BM25sDocument(id="doc2", content="This is the second document"),
            BM25sDocument(id="doc3", content="Another document with different content")
        ]
    
    @patch('refinire_rag_bm25s_j.services.bm25s')
    def test_create_index(self, mock_bm25s):
        """Test index creation."""
        mock_index = Mock()
        mock_bm25s.BM25.return_value = mock_index
        mock_bm25s.tokenize.return_value = [["this", "is"], ["the", "first"]]
        
        self.service.create_index(self.test_documents)
        
        assert self.service.index == mock_index
        assert len(self.service._documents) == 3
        mock_index.index.assert_called_once()
    
    def test_create_index_empty_documents(self):
        """Test creating index with empty documents list."""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            self.service.create_index([])
    
    @patch('refinire_rag_bm25s_j.services.pickle')
    def test_save_and_load_index(self, mock_pickle):
        """Test saving and loading index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test_index.pkl"
            self.config.index_path = str(index_path)
            
            service = BM25sIndexService(self.config)
            
            # Mock the index for saving
            mock_index = Mock()
            service.index = mock_index
            service._documents = self.test_documents
            
            # Save index
            service.save_index()
            assert index_path.exists()
            mock_pickle.dump.assert_called_once()
            
            # Test load with mocked pickle
            mock_pickle.load.return_value = {
                'index': mock_index,
                'documents': self.test_documents
            }
            
            # Load index
            new_service = BM25sIndexService(self.config)
            new_service.load_index()
            
            assert len(new_service._documents) == 3
    
    def test_load_index_file_not_found(self):
        """Test loading index when file doesn't exist."""
        self.config.index_path = "/nonexistent/path.pkl"
        
        with pytest.raises(FileNotFoundError):
            self.service.load_index()
    
    def test_save_index_no_path(self):
        """Test saving index without configured path."""
        with pytest.raises(ValueError, match="Index path not configured"):
            self.service.save_index()


class TestBM25sSearchService:
    """Test BM25sSearchService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BM25sConfig()
        self.index_service = BM25sIndexService(self.config)
        self.search_service = BM25sSearchService(self.index_service)
        
        self.test_documents = [
            BM25sDocument(id="doc1", content="Python programming language"),
            BM25sDocument(id="doc2", content="Machine learning with Python"),
            BM25sDocument(id="doc3", content="Data science and analytics")
        ]
    
    @patch('refinire_rag_bm25s_j.services.bm25s')
    def test_search(self, mock_bm25s):
        """Test search functionality."""
        # Mock index setup
        mock_index = Mock()
        mock_index.retrieve.return_value = ([[0.9, 0.7]], [[0, 1]])
        self.index_service.index = mock_index
        self.index_service._documents = self.test_documents
        
        mock_bm25s.tokenize.return_value = ["python"]
        
        results = self.search_service.search("python", top_k=2)
        
        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[0].rank == 1
        assert results[1].score == 0.7
        assert results[1].rank == 2
    
    def test_search_no_index(self):
        """Test search without available index."""
        with pytest.raises(ValueError, match="Index not available"):
            self.search_service.search("test query")
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        self.index_service.index = Mock()
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.search_service.search("")
    
    @patch('refinire_rag_bm25s_j.services.bm25s')
    def test_batch_search(self, mock_bm25s):
        """Test batch search functionality."""
        # Mock index setup
        mock_index = Mock()
        mock_index.retrieve.return_value = ([[0.9]], [[0]])
        self.index_service.index = mock_index
        self.index_service._documents = self.test_documents
        
        mock_bm25s.tokenize.return_value = ["python"]
        
        queries = ["python", "machine learning"]
        results = self.search_service.batch_search(queries, top_k=1)
        
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1
    
    def test_batch_search_empty_queries(self):
        """Test batch search with empty queries list."""
        with pytest.raises(ValueError, match="Queries list cannot be empty"):
            self.search_service.batch_search([])
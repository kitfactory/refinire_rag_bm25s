"""Unit tests for BM25sKeywordStore."""

import os
import pytest
from unittest.mock import Mock, patch
from typing import List

from refinire_rag.models.document import Document
from refinire_rag.retrieval.base import SearchResult

from refinire_rag_bm25s_j.keyword_store import BM25sKeywordStore
from refinire_rag_bm25s_j.models import BM25sConfig, BM25sDocument


class TestBM25sKeywordStore:
    """Test cases for BM25sKeywordStore."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "index_path": None,
            "k1": 1.2,
            "b": 0.75,
            "epsilon": 0.25
        }
        self.keyword_store = BM25sKeywordStore(config=self.config)
    
    def test_initialization(self):
        """Test keyword store initialization."""
        assert self.keyword_store is not None
        assert isinstance(self.keyword_store.bm25s_config, BM25sConfig)
        assert self.keyword_store.bm25s_config.k1 == 1.2
        assert self.keyword_store.bm25s_config.b == 0.75
        assert self.keyword_store.bm25s_config.epsilon == 0.25
    
    def test_initialization_with_bm25s_config(self):
        """Test initialization with BM25sConfig object."""
        config = BM25sConfig(k1=1.5, b=0.8)
        store = BM25sKeywordStore(config=config)
        assert store.bm25s_config.k1 == 1.5
        assert store.bm25s_config.b == 0.8
    
    def test_get_config_class(self):
        """Test get_config_class method."""
        config_class = BM25sKeywordStore.get_config_class()
        from typing import Dict
        assert config_class == Dict
    
    def test_get_config(self):
        """Test get_config method."""
        store = BM25sKeywordStore(config=self.config)
        config_dict = store.get_config()
        
        assert isinstance(config_dict, dict)
        assert config_dict["k1"] == 1.2
        assert config_dict["b"] == 0.75
        assert config_dict["epsilon"] == 0.25
    
    def test_add_document(self):
        """Test adding a single document."""
        doc = Document(
            id="doc1",
            content="これはテストドキュメントです。",
            metadata={"category": "test"}
        )
        
        with patch.object(self.keyword_store, 'index_documents') as mock_index:
            self.keyword_store.add_document(doc)
            mock_index.assert_called_once_with([doc])
    
    def test_index_document(self):
        """Test indexing a single document."""
        doc = Document(
            id="doc1",
            content="これはテストドキュメントです。",
            metadata={"category": "test"}
        )
        
        with patch.object(self.keyword_store, 'index_documents') as mock_index:
            self.keyword_store.index_document(doc)
            mock_index.assert_called_once_with([doc])
    
    @patch('refinire_rag_bm25s_j.keyword_store.BM25sIndexService')
    def test_index_documents(self, mock_service_class):
        """Test indexing multiple documents."""
        mock_service = Mock()
        mock_service.get_documents.return_value = []
        mock_service_class.return_value = mock_service
        
        store = BM25sKeywordStore(config=self.config)
        
        docs = [
            Document(id="doc1", content="テスト1"),
            Document(id="doc2", content="テスト2")
        ]
        
        store.index_documents(docs)
        
        # Verify create_index was called with BM25sDocument objects
        mock_service.create_index.assert_called_once()
        args = mock_service.create_index.call_args[0][0]
        assert len(args) == 2
        assert all(isinstance(doc, BM25sDocument) for doc in args)
    
    @patch('refinire_rag_bm25s_j.keyword_store.BM25sIndexService')
    def test_remove_document(self, mock_service_class):
        """Test removing a document."""
        mock_service = Mock()
        mock_doc = BM25sDocument(id="doc1", content="test", metadata={})
        mock_service.get_documents.return_value = [mock_doc]
        mock_service_class.return_value = mock_service
        
        store = BM25sKeywordStore(config=self.config)
        
        result = store.remove_document("doc1")
        
        assert result is True
        mock_service.create_index.assert_called_once_with([])
    
    @patch('refinire_rag_bm25s_j.keyword_store.BM25sIndexService')
    def test_remove_nonexistent_document(self, mock_service_class):
        """Test removing a nonexistent document."""
        mock_service = Mock()
        mock_doc = BM25sDocument(id="doc1", content="test", metadata={})
        mock_service.get_documents.return_value = [mock_doc]
        mock_service_class.return_value = mock_service
        
        store = BM25sKeywordStore(config=self.config)
        
        result = store.remove_document("nonexistent")
        
        assert result is False
        mock_service.create_index.assert_not_called()
    
    @patch('refinire_rag_bm25s_j.keyword_store.BM25sIndexService')
    def test_update_document(self, mock_service_class):
        """Test updating a document."""
        mock_service = Mock()
        original_doc = BM25sDocument(id="doc1", content="original", metadata={})
        mock_service.get_documents.return_value = [original_doc]
        mock_service_class.return_value = mock_service
        
        store = BM25sKeywordStore(config=self.config)
        
        updated_doc = Document(
            id="doc1",
            content="updated content",
            metadata={"updated": True}
        )
        
        result = store.update_document(updated_doc)
        
        assert result is True
        mock_service.create_index.assert_called_once()
    
    @patch('refinire_rag_bm25s_j.keyword_store.BM25sIndexService')
    def test_update_nonexistent_document(self, mock_service_class):
        """Test updating a nonexistent document."""
        mock_service = Mock()
        mock_service.get_documents.return_value = []
        mock_service_class.return_value = mock_service
        
        store = BM25sKeywordStore(config=self.config)
        
        doc = Document(id="nonexistent", content="test")
        result = store.update_document(doc)
        
        assert result is False
        mock_service.create_index.assert_not_called()
    
    def test_clear_index(self):
        """Test clearing the index."""
        self.keyword_store.clear_index()
        
        assert self.keyword_store.index_service.index is None
        assert self.keyword_store.index_service._documents == []
    
    @patch('refinire_rag_bm25s_j.keyword_store.BM25sIndexService')
    def test_get_document_count(self, mock_service_class):
        """Test getting document count."""
        mock_service = Mock()
        mock_service.get_documents.return_value = [Mock(), Mock(), Mock()]
        mock_service_class.return_value = mock_service
        
        store = BM25sKeywordStore(config=self.config)
        
        count = store.get_document_count()
        assert count == 3
    
    @patch('refinire_rag_bm25s_j.keyword_store.BM25sSearchService')
    def test_retrieve(self, mock_search_service_class):
        """Test retrieving documents."""
        mock_search_service = Mock()
        mock_result = Mock()
        mock_result.document = BM25sDocument(
            id="doc1", 
            content="テストコンテンツ", 
            metadata={"category": "test"}
        )
        mock_result.score = 0.85
        mock_result.rank = 1
        
        mock_search_service.search.return_value = [mock_result]
        mock_search_service_class.return_value = mock_search_service
        
        store = BM25sKeywordStore(config=self.config)
        
        results = store.retrieve("テスト", limit=5)
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].document.content == "テストコンテンツ"
        assert results[0].score == 0.85
        # refinire-rag SearchResult doesn't have a rank attribute
        assert results[0].document_id == "doc1"
        # Check that the original metadata is included (refinire-rag Document adds required fields)
        assert "category" in results[0].metadata
        assert results[0].metadata["category"] == "test"
        
        mock_search_service.search.assert_called_once_with(
            "テスト", top_k=5, metadata_filter=None
        )
    
    @patch('refinire_rag_bm25s_j.keyword_store.BM25sSearchService')
    def test_search(self, mock_search_service_class):
        """Test search method."""
        mock_search_service = Mock()
        mock_result = Mock()
        mock_result.document = BM25sDocument(id="doc1", content="test", metadata={})
        mock_result.score = 0.9
        mock_result.rank = 1
        
        mock_search_service.search.return_value = [mock_result]
        mock_search_service_class.return_value = mock_search_service
        
        store = BM25sKeywordStore(config=self.config)
        
        results = store.search("test", limit=3)
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        mock_search_service.search.assert_called_once_with(
            "test", top_k=3, metadata_filter=None
        )
    
    def test_process(self):
        """Test process method (DocumentProcessor interface)."""
        docs = [
            Document(id="1", content="doc1"),
            Document(id="2", content="doc2")
        ]
        
        with patch.object(self.keyword_store, 'index_documents') as mock_index:
            result_docs = self.keyword_store.process(docs)
            
            assert result_docs == docs  # Should return unchanged
            mock_index.assert_called_once_with(docs)
    
    def test_retrieve_with_metadata_filter(self):
        """Test retrieve with metadata filter."""
        with patch.object(self.keyword_store.search_service, 'search') as mock_search:
            mock_search.return_value = []
            
            self.keyword_store.retrieve(
                "test query", 
                limit=5, 
                metadata_filter={"category": "tech"}
            )
            
            mock_search.assert_called_once_with(
                "test query", 
                top_k=5, 
                metadata_filter={"category": "tech"}
            )
    
    def test_initialization_with_kwargs(self):
        """Test initialization with kwargs parameters."""
        store = BM25sKeywordStore(
            k1=1.8,
            b=0.9,
            index_path="/custom/path",
            method="bm25+"
        )
        
        assert store.bm25s_config.k1 == 1.8
        assert store.bm25s_config.b == 0.9
        assert store.bm25s_config.index_path == "/custom/path"
        assert store.bm25s_config.method == "bm25+"
    
    @patch.dict(os.environ, {
        "REFINIRE_RAG_BM25S_K1": "2.0",
        "REFINIRE_RAG_BM25S_B": "0.85",
        "REFINIRE_RAG_BM25S_INDEX_PATH": "/env/path"
    })
    def test_initialization_with_env_vars(self):
        """Test initialization with environment variables."""
        store = BM25sKeywordStore()
        
        assert store.bm25s_config.k1 == 2.0
        assert store.bm25s_config.b == 0.85
        assert store.bm25s_config.index_path == "/env/path"
    
    @patch.dict(os.environ, {
        "REFINIRE_RAG_BM25S_K1": "2.0",
        "REFINIRE_RAG_BM25S_B": "0.85"
    })
    def test_kwargs_override_env_vars(self):
        """Test that kwargs override environment variables."""
        store = BM25sKeywordStore(
            k1=2.5,  # Should override env var
            epsilon=0.5  # Should use this value
        )
        
        assert store.bm25s_config.k1 == 2.5  # kwargs override
        assert store.bm25s_config.b == 0.85  # from env var
        assert store.bm25s_config.epsilon == 0.5  # from kwargs
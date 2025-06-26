"""Tests for metadata filtering functionality."""

from unittest.mock import Mock, patch
import pytest

from refinire_rag_bm25s_j.models import BM25sConfig, BM25sDocument
from refinire_rag_bm25s_j.services import BM25sIndexService, BM25sSearchService


class TestMetadataFiltering:
    """Test metadata filtering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BM25sConfig()
        self.index_service = BM25sIndexService(self.config)
        self.search_service = BM25sSearchService(self.index_service)
        
        self.test_documents = [
            BM25sDocument(
                id="doc1", 
                content="Python programming tutorial",
                metadata={"category": "tech", "language": "python", "difficulty": "beginner"}
            ),
            BM25sDocument(
                id="doc2", 
                content="Advanced machine learning algorithms",
                metadata={"category": "tech", "language": "python", "difficulty": "advanced"}
            ),
            BM25sDocument(
                id="doc3", 
                content="Business financial report",
                metadata={"category": "business", "type": "report", "year": 2024}
            ),
            BM25sDocument(
                id="doc4", 
                content="Legal document analysis",
                metadata={"category": "legal", "type": "analysis", "year": 2023}
            )
        ]
    
    def test_matches_filter_exact_match(self):
        """Test exact match filtering."""
        metadata = {"category": "tech", "language": "python"}
        filter_dict = {"category": "tech"}
        
        result = self.search_service._matches_filter(metadata, filter_dict)
        assert result is True
        
        # Test non-matching
        filter_dict = {"category": "business"}
        result = self.search_service._matches_filter(metadata, filter_dict)
        assert result is False
    
    def test_matches_filter_multiple_criteria(self):
        """Test filtering with multiple criteria."""
        metadata = {"category": "tech", "language": "python", "difficulty": "beginner"}
        
        # All criteria match
        filter_dict = {"category": "tech", "language": "python"}
        result = self.search_service._matches_filter(metadata, filter_dict)
        assert result is True
        
        # One criterion doesn't match
        filter_dict = {"category": "tech", "language": "java"}
        result = self.search_service._matches_filter(metadata, filter_dict)
        assert result is False
    
    def test_matches_filter_list_values(self):
        """Test filtering with list values (OR condition)."""
        metadata = {"category": "tech"}
        
        # Category in list
        filter_dict = {"category": ["tech", "business"]}
        result = self.search_service._matches_filter(metadata, filter_dict)
        assert result is True
        
        # Category not in list
        filter_dict = {"category": ["business", "legal"]}
        result = self.search_service._matches_filter(metadata, filter_dict)
        assert result is False
    
    def test_matches_filter_empty_metadata(self):
        """Test filtering with empty metadata."""
        metadata = None
        
        # Empty filter should match empty metadata
        filter_dict = {}
        result = self.search_service._matches_filter(metadata, filter_dict)
        assert result is True
        
        # Non-empty filter should not match empty metadata
        filter_dict = {"category": "tech"}
        result = self.search_service._matches_filter(metadata, filter_dict)
        assert result is False
    
    def test_apply_operator_filter_comparison(self):
        """Test comparison operators."""
        # Greater than
        result = self.search_service._apply_operator_filter(5, {"$gt": 3})
        assert result is True
        
        result = self.search_service._apply_operator_filter(2, {"$gt": 3})
        assert result is False
        
        # Greater than or equal
        result = self.search_service._apply_operator_filter(3, {"$gte": 3})
        assert result is True
        
        # Less than
        result = self.search_service._apply_operator_filter(2, {"$lt": 3})
        assert result is True
        
        # Less than or equal
        result = self.search_service._apply_operator_filter(3, {"$lte": 3})
        assert result is True
    
    def test_apply_operator_filter_in_operations(self):
        """Test $in and $nin operators."""
        # $in operator
        result = self.search_service._apply_operator_filter("python", {"$in": ["python", "java"]})
        assert result is True
        
        result = self.search_service._apply_operator_filter("rust", {"$in": ["python", "java"]})
        assert result is False
        
        # $nin operator
        result = self.search_service._apply_operator_filter("rust", {"$nin": ["python", "java"]})
        assert result is True
        
        result = self.search_service._apply_operator_filter("python", {"$nin": ["python", "java"]})
        assert result is False
    
    def test_apply_operator_filter_not_equal(self):
        """Test $ne operator."""
        result = self.search_service._apply_operator_filter("python", {"$ne": "java"})
        assert result is True
        
        result = self.search_service._apply_operator_filter("python", {"$ne": "python"})
        assert result is False
    
    def test_apply_operator_filter_exists(self):
        """Test $exists operator."""
        # Field exists
        result = self.search_service._apply_operator_filter("value", {"$exists": True})
        assert result is True
        
        result = self.search_service._apply_operator_filter(None, {"$exists": True})
        assert result is False
        
        # Field doesn't exist
        result = self.search_service._apply_operator_filter(None, {"$exists": False})
        assert result is True
        
        result = self.search_service._apply_operator_filter("value", {"$exists": False})
        assert result is False
    
    def test_apply_operator_filter_unsupported(self):
        """Test unsupported operator (should be ignored)."""
        result = self.search_service._apply_operator_filter("value", {"$unsupported": "test"})
        assert result is True  # Should ignore unsupported operators
    
    def test_apply_operator_filter_none_value(self):
        """Test operator filtering with None values."""
        # None value with comparison operators should return False
        result = self.search_service._apply_operator_filter(None, {"$gt": 5})
        assert result is False
        
        result = self.search_service._apply_operator_filter(None, {"$gte": 5})
        assert result is False
    
    @patch('refinire_rag_bm25s_j.services.bm25s')
    def test_search_with_metadata_filter_native_support(self, mock_bm25s):
        """Test search with metadata filter when native support is available."""
        # Mock native filtering support
        mock_index = Mock()
        mock_index.retrieve.return_value = ([[0.9]], [[0]])
        self.index_service.index = mock_index
        self.index_service._documents = self.test_documents
        
        # Mock tokenize
        mock_bm25s.tokenize.return_value = ["python"]
        
        # Mock supports_native_filtering to return True
        with patch.object(self.search_service, '_supports_native_filtering', return_value=True):
            filter_dict = {"category": "tech"}
            results = self.search_service.search("python", top_k=2, metadata_filter=filter_dict)
            
            # Should call retrieve with filter parameter
            mock_index.retrieve.assert_called_with(["python"], k=2, filter=filter_dict)
            assert len(results) == 1
    
    @patch('refinire_rag_bm25s_j.services.bm25s')
    def test_search_with_metadata_filter_fallback(self, mock_bm25s):
        """Test search with metadata filter using fallback method."""
        # Mock index without native filtering support
        mock_index = Mock()
        mock_index.retrieve.side_effect = [
            # First call with filter fails
            TypeError("filter parameter not supported"),
            # Second call without filter succeeds
            ([[0.9, 0.7, 0.5]], [[0, 1, 2]])
        ]
        self.index_service.index = mock_index
        self.index_service._documents = self.test_documents
        
        mock_bm25s.tokenize.return_value = ["python"]
        
        # Mock supports_native_filtering to return False
        with patch.object(self.search_service, '_supports_native_filtering', return_value=False):
            filter_dict = {"category": "tech"}
            results = self.search_service.search("python", top_k=2, metadata_filter=filter_dict)
            
            # Should apply post-filtering
            assert len(results) <= 2
            # All results should match the filter
            for result in results:
                assert result.document.metadata.get("category") == "tech"
    
    @patch('refinire_rag_bm25s_j.services.bm25s')
    def test_search_without_metadata_filter(self, mock_bm25s):
        """Test search without metadata filter."""
        mock_index = Mock()
        mock_index.retrieve.return_value = ([[0.9, 0.7]], [[0, 1]])
        self.index_service.index = mock_index
        self.index_service._documents = self.test_documents
        
        mock_bm25s.tokenize.return_value = ["python"]
        
        results = self.search_service.search("python", top_k=2)
        
        # Should call retrieve without filter
        mock_index.retrieve.assert_called_with(["python"], k=2)
        assert len(results) == 2
    
    @patch('refinire_rag_bm25s_j.services.bm25s')
    def test_batch_search_with_metadata_filter(self, mock_bm25s):
        """Test batch search with metadata filter."""
        mock_index = Mock()
        mock_index.retrieve.return_value = ([[0.9]], [[0]])
        self.index_service.index = mock_index
        self.index_service._documents = self.test_documents
        
        mock_bm25s.tokenize.return_value = ["test"]
        
        with patch.object(self.search_service, '_supports_native_filtering', return_value=True):
            queries = ["python programming", "machine learning"]
            filter_dict = {"category": "tech"}
            
            results = self.search_service.batch_search(
                queries, top_k=2, metadata_filter=filter_dict
            )
            
            assert len(results) == 2  # Two queries
            assert all(isinstance(query_results, list) for query_results in results)
    
    def test_supports_native_filtering_detection(self):
        """Test detection of native filtering support."""
        # Mock index with filter parameter
        mock_index = Mock()
        
        # Create a mock retrieve method with filter parameter
        def mock_retrieve(query, k, filter=None):
            return ([], [])
        
        mock_index.retrieve = mock_retrieve
        self.index_service.index = mock_index
        
        # Should detect filter parameter
        supports_filtering = self.search_service._supports_native_filtering()
        assert supports_filtering is True
    
    def test_supports_native_filtering_no_support(self):
        """Test detection when native filtering is not supported."""
        # Mock index without filter parameter
        mock_index = Mock()
        
        def mock_retrieve(query, k):
            return ([], [])
        
        mock_index.retrieve = mock_retrieve
        self.index_service.index = mock_index
        
        # Should not detect filter parameter
        supports_filtering = self.search_service._supports_native_filtering()
        assert supports_filtering is False
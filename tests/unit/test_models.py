"""Tests for data models."""

import pytest
from pydantic import ValidationError

from refinire_rag_bm25s_j.models import BM25sConfig, BM25sDocument


class TestBM25sDocument:
    """Test BM25sDocument class."""
    
    def test_valid_document(self):
        """Test creating a valid document."""
        doc = BM25sDocument(
            id="test-1",
            content="This is a test document",
            metadata={"author": "test"}
        )
        assert doc.validate()
        assert doc.id == "test-1"
        assert doc.content == "This is a test document"
        assert doc.metadata["author"] == "test"
    
    def test_document_with_none_metadata(self):
        """Test document with None metadata."""
        doc = BM25sDocument(id="test-1", content="Test content")
        assert doc.validate()
        assert doc.metadata == {}
    
    def test_invalid_document_empty_id(self):
        """Test document with empty ID."""
        doc = BM25sDocument(id="", content="Test content")
        assert not doc.validate()
    
    def test_invalid_document_empty_content(self):
        """Test document with empty content."""
        doc = BM25sDocument(id="test-1", content="")
        assert not doc.validate()


class TestBM25sConfig:
    """Test BM25sConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = BM25sConfig()
        assert config.k1 == 1.2
        assert config.b == 0.75
        assert config.epsilon == 0.25
        assert config.index_path is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BM25sConfig(
            k1=1.5,
            b=0.8,
            epsilon=0.3,
            index_path="/tmp/test_index.pkl"
        )
        assert config.k1 == 1.5
        assert config.b == 0.8
        assert config.epsilon == 0.3
        assert config.index_path == "/tmp/test_index.pkl"
    
    def test_config_validation_k1_negative(self):
        """Test k1 validation with negative value."""
        with pytest.raises(ValidationError):
            BM25sConfig(k1=-0.1)
    
    def test_config_validation_b_out_of_range(self):
        """Test b validation with out of range value."""
        with pytest.raises(ValidationError):
            BM25sConfig(b=1.5)
        
        with pytest.raises(ValidationError):
            BM25sConfig(b=-0.1)
    
    def test_to_dict(self):
        """Test to_dict method."""
        config = BM25sConfig(k1=1.5, b=0.8)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["k1"] == 1.5
        assert config_dict["b"] == 0.8
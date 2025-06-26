"""Data models for BM25s VectorStore."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


@dataclass
class BM25sDocument:
    """Document entity for BM25s processing."""
    
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def validate(self) -> bool:
        """Validate document data."""
        if not self.id or not self.content:
            return False
        if self.metadata is None:
            self.metadata = {}
        return True


class BM25sConfig(BaseModel):
    """Configuration for BM25s indexing and search."""
    
    k1: float = Field(default=1.2, ge=0.0, description="Term frequency saturation parameter")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="Length normalization parameter")
    epsilon: float = Field(default=0.25, ge=0.0, description="IDF cutoff parameter")
    index_path: Optional[str] = Field(default=None, description="Path to save/load index")
    
    @field_validator('k1')
    @classmethod
    def validate_k1(cls, v):
        if v < 0:
            raise ValueError('k1 must be non-negative')
        return v
    
    @field_validator('b')
    @classmethod
    def validate_b(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('b must be between 0 and 1')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()



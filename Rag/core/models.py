"""
Pydantic models for structured RAG responses with citations.
"""

from typing import List
from pydantic import BaseModel


class Citation(BaseModel):
    """A single citation linking an answer back to a source document."""
    document_name: str
    page_number: int
    retrieved_text: str
    similarity_score: float


class QueryResult(BaseModel):
    """Complete result of a RAG query with answer and supporting citations."""
    question: str
    answer: str
    citations: List[Citation]
    is_answerable: bool = True

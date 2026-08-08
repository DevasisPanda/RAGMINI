"""
Rag - A minimal Retrieval-Augmented Generation library with structured responses and source citations
"""

from .core.provider import Provider
from .core.rag_engine import RagEngine
from .core.models import Citation, QueryResult

__version__ = "2.0.0"
__all__ = ["Provider", "RagEngine", "Citation", "QueryResult"]
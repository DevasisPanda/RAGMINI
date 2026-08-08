"""
TinyRag - A minimal Retrieval-Augmented Generation library with structured responses and source citations
"""

from .core.provider import Provider
from .core.tinyrag import TinyRag
from .core.models import Citation, QueryResult

__version__ = "1.0.0"
__all__ = ["Provider", "TinyRag", "Citation", "QueryResult"]
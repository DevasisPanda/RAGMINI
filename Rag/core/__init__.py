"""
Rag - A minimal Retrieval-Augmented Generation library
"""

from .provider import Provider
from .rag_engine import RagEngine

__version__ = "2.0.0"
__all__ = ["Provider", "RagEngine"]
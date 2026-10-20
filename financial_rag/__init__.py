"""
financial_rag package initialization.
Exposes core components for the Financial RAG system.
"""

from financial_rag.rag_pipeline import RagPipeline
from financial_rag.generator import generate, generate_stream
from financial_rag.retriever import FinancialRetriever, build_retriever
from financial_rag.reranker import RerankingRetriever

__all__ = [
    "RagPipeline",
    "generate",
    "generate_stream",
    "FinancialRetriever",
    "build_retriever",
    "RerankingRetriever",
]

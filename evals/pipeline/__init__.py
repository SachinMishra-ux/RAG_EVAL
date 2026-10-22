"""
evals/pipeline
==============
Pipeline-level evaluations measure the live, integrated RAG chain (Retriever + Generator together).
Evaluates the classical RAG Triad:
  1. Contextual Relevancy
  2. Faithfulness
  3. Answer Relevancy

Available modules:
  - eval_rag_pipeline : Live system pipeline evaluation
"""

from evals.pipeline import eval_rag_pipeline

__all__ = [
    "eval_rag_pipeline",
]

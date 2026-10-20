"""
financial_rag/reranker.py
=========================
Reranker component using a Cross-Encoder.
Two-stage retrieval architecture:
  1. Over-retrieve: Fast bi-encoder vector search retrieves fetch_k candidates.
  2. Rerank: Cross-encoder jointly scores (query, chunk) pairs for deep lexical + semantic ranking.
  3. Filter: Returns top_k most relevant documents.

Usage:
    from financial_rag.reranker import RerankingRetriever
    retriever = RerankingRetriever(fetch_k=10, top_k=5)
    docs = retriever.invoke("What was the ROTCE for JPMorgan?")
"""

from sentence_transformers import CrossEncoder
from langsmith import traceable
from financial_rag.retriever import build_retriever

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    def __init__(self, fetch_k: int = 10, top_k: int = 5):
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.base_retriever = build_retriever(k=fetch_k)
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)

    @traceable(run_type="retriever", name="FinancialRerankingRetriever")
    def invoke(self, query: str):
        # 1. OVER-RETRIEVE: fast vector bi-encoder
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        # 2. RERANK: score each (query, chunk) pair together
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 3. SORT by cross-encoder score in descending order
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]


if __name__ == "__main__":
    rr = RerankingRetriever(fetch_k=6, top_k=3)
    results = rr.invoke("total assets and deposits")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.page_content[:120]}...\n")

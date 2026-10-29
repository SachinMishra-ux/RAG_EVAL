"""
financial_rag/rag_pipeline.py
=============================
The unified RAG Pipeline orchestrator.
Coordinates:
  1. Retrieval: Over-retrieves candidates via vector similarity.
  2. Reranking: Scores candidates with a Cross-Encoder down to top_k.
  3. Generation: Produces a strictly grounded answer from the context chunks.

Returns the complete RAG Triad:
  - query: the user prompt
  - context: list of retrieved string chunks
  - answer: the generated grounded response
"""

from langsmith import traceable
from financial_rag.reranker import RerankingRetriever
from financial_rag.generator import generate


class RagPipeline:
    def __init__(self, fetch_k: int = 10, top_k: int = 5):
        # Build retriever once (reused across queries and evaluations)
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)

    @traceable(run_type="chain", name="FinancialRagPipeline")
    def invoke(self, query: str, llm_override=None) -> dict:
        # 1. RETRIEVE: over-fetch then rerank down to top_k documents
        docs = self.retriever.invoke(query)

        # 2. UNPACK: extract list of string chunks for generator and DeepEval metrics
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: produce grounded answer from the retrieved context
        answer = generate(query, context, llm_override=llm_override)

        # Return all legs of the RAG triad
        return {
            "query": query,
            "context": context,
            "answer": answer,
            "documents": docs,
        }

    def close(self):
        """Releases vector store client and any open resources cleanly."""
        try:
            if hasattr(self, "retriever"):
                base_retriever = getattr(self.retriever, "base_retriever", None)
                if base_retriever and hasattr(base_retriever, "qdrant_manager"):
                    base_retriever.qdrant_manager.close()
        except Exception:
            pass


if __name__ == "__main__":
    rag = RagPipeline(fetch_k=6, top_k=3)
    res = rag.invoke("What was JPMorgan's revenue and net income in 2023?")
    print("QUERY:  ", res["query"])
    print("ANSWER: ", res["answer"])
    print("\nRETRIEVED CHUNKS:")
    for i, chunk in enumerate(res["context"], 1):
        print(f"  [{i}] {chunk[:100]}...")

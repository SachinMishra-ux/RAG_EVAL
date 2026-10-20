"""
financial_rag/retriever.py
==========================
Component 1: The RETRIEVER component.
Extracts financial PDF transcripts / disclosures, chunks them, embeds them,
and indexes them in Qdrant for semantic similarity search.

Usage:
    from financial_rag.retriever import build_retriever
    retriever = build_retriever(k=5)
    docs = retriever.invoke("What was JPMorgan's net income in 2023?")
"""

import os
import glob
from dotenv import load_dotenv
from langchain_core.documents import Document

from financial_rag.document_processor import extract_text_from_pdf, chunk_documents
from financial_rag.embedder import get_default_embedder
from financial_rag.qdrant_manager import QdrantManager

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
COLLECTION_NAME = "financial_documents"

_global_store = None


def load_store() -> QdrantManager:
    """
    Initializes and returns the Qdrant store.
    If the store is empty, automatically reads and indexes PDFs from data/.
    """
    global _global_store
    if _global_store is not None:
        return _global_store

    store = QdrantManager(collection_name=COLLECTION_NAME)
    embedder = get_default_embedder()

    # Check if collection is already populated
    existing_records = store.get_all_records(limit=1)
    if not existing_records:
        print("[Retriever] Initializing vector store from financial reports...")
        pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
        if not pdf_paths:
            print(f"[Retriever] Warning: No PDF files found in {DATA_DIR}.")
            _global_store = store
            return store

        all_chunks = []
        for path in pdf_paths:
            print(f"[Retriever] Processing {os.path.basename(path)}...")
            docs = extract_text_from_pdf(path, max_pages=60)  # fast, comprehensive seed sample
            chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=150)
            all_chunks.extend(chunks)

        if all_chunks:
            sample_emb = embedder.embed_text(all_chunks[0].page_content)
            vector_size = len(sample_emb)
            store.ensure_collection(vector_size=vector_size)

            embeddings = []
            payloads = []
            print(f"[Retriever] Generating embeddings for {len(all_chunks)} chunks...")
            for idx, c in enumerate(all_chunks):
                if idx == 0:
                    emb = sample_emb
                else:
                    emb = embedder.embed_text(c.page_content)
                embeddings.append(emb)
                payloads.append({
                    "text": c.page_content,
                    "source": c.metadata.get("source", "unknown"),
                    "page": c.metadata.get("page", 1),
                })

            store.insert_records(embeddings, payloads)

    _global_store = store
    return store


class FinancialRetriever:
    """
    LangChain-compatible retriever class wrapping Qdrant vector search.
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.store = load_store()
        self.embedder = get_default_embedder()

    def invoke(self, query: str) -> list[Document]:
        """Embeds query text and returns top-k matching documents from Qdrant."""
        query_vector = self.embedder.embed_text(query)
        return self.store.search(query_vector, limit=self.k)


def build_retriever(k: int = 5) -> FinancialRetriever:
    return FinancialRetriever(k=k)


if __name__ == "__main__":
    retriever = build_retriever(k=3)
    results = retriever.invoke("What was JPMorgan's net income in 2023?")
    print(f"\nRetrieved {len(results)} chunks:")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] Page {doc.metadata.get('page')}: {doc.page_content[:140]}...\n")

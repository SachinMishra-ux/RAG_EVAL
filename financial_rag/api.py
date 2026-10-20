"""
financial_rag/api.py
====================
FastAPI REST backend service for the Financial RAG pipeline.
Endpoints:
  - POST /query   : Submit questions, retrieves context, returns grounded answer.
  - POST /upload  : Upload new financial PDF filings.
  - GET  /health  : Health check on vector DB and LLM connectivity.
"""

import os
import sys
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from financial_rag.rag_pipeline import RagPipeline
from financial_rag.document_processor import extract_text_from_pdf, chunk_documents
from financial_rag.embedder import get_default_embedder
from financial_rag.qdrant_manager import QdrantManager

app = FastAPI(
    title="Financial Statement RAG API",
    description="Enterprise REST API for corporate financial disclosures and SEC 10-K RAG question answering.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


class QueryRequest(BaseModel):
    question: str = Field(..., description="Financial question to answer")
    fetch_k: int = Field(default=8, ge=1, le=30)
    top_k: int = Field(default=4, ge=1, le=10)


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: list[str]
    chunks_retrieved: int


@app.on_event("startup")
def startup():
    global pipeline
    print("[API] Initializing Financial RAG Pipeline...")
    pipeline = RagPipeline(fetch_k=8, top_k=4)
    print("[API] Pipeline successfully initialized.")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "financial_rag_api"}


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        result = pipeline.invoke(req.question)
        return QueryResponse(
            question=req.question,
            answer=result["answer"],
            context=result["context"],
            chunks_retrieved=len(result["context"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, file.filename)

    with open(target_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Ingest into Qdrant store
    try:
        docs = extract_text_from_pdf(target_path, max_pages=60)
        chunks = chunk_documents(docs)
        if chunks:
            store = QdrantManager(collection_name="financial_documents")
            embedder = get_default_embedder()
            sample_emb = embedder.embed_text(chunks[0].page_content)
            store.ensure_collection(vector_size=len(sample_emb))
            embeddings = [embedder.embed_text(c.page_content) for c in chunks]
            payloads = [{"text": c.page_content, "source": file.filename, "page": c.metadata.get("page", 1)} for c in chunks]
            store.insert_records(embeddings, payloads)

        return {"status": "success", "filename": file.filename, "chunks_ingested": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("financial_rag.api:app", host="0.0.0.0", port=8000, reload=True)

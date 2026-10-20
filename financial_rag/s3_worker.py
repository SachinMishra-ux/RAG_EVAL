"""
financial_rag/s3_worker.py
==========================
AWS S3 & SQS Event-Driven Document Ingestion Worker.
Listens to SQS queue for S3 ObjectCreated events, downloads incoming financial
PDF filings, chunks, embeds, and loads them into Qdrant vector database.
"""

import os
import sys
import json
import time
import tempfile
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from financial_rag.document_processor import extract_text_from_pdf, chunk_documents
from financial_rag.embedder import get_default_embedder
from financial_rag.qdrant_manager import QdrantManager

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financial_documents")


def start_worker():
    if not SQS_QUEUE_URL:
        print("[S3 Worker] SQS_QUEUE_URL not configured. Worker exiting.")
        return

    import boto3
    from botocore.exceptions import ClientError

    sqs = boto3.client("sqs", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    store = QdrantManager(collection_name=COLLECTION_NAME)
    embedder = get_default_embedder()

    print(f"🚀 S3 Ingestion Worker active on queue: {SQS_QUEUE_URL}")
    while True:
        try:
            res = sqs.receive_message(QueueUrl=SQS_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)
            messages = res.get("Messages", [])
            for msg in messages:
                body = json.loads(msg["Body"])
                records = body.get("Records", [])
                for record in records:
                    bucket = record["s3"]["bucket"]["name"]
                    key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
                    if not key.lower().endswith(".pdf"):
                        continue

                    print(f"[S3 Worker] Downloading s3://{bucket}/{key}...")
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp_path = tmp.name
                    s3.download_file(bucket, key, tmp_path)

                    docs = extract_text_from_pdf(tmp_path, max_pages=60)
                    chunks = chunk_documents(docs)
                    if chunks:
                        sample_emb = embedder.embed_text(chunks[0].page_content)
                        store.ensure_collection(vector_size=len(sample_emb))
                        embeddings = [embedder.embed_text(c.page_content) for c in chunks]
                        payloads = [{"text": c.page_content, "source": key, "page": c.metadata.get("page", 1)} for c in chunks]
                        store.insert_records(embeddings, payloads)

                    os.remove(tmp_path)
                    print(f"[S3 Worker] Ingested s3://{bucket}/{key}")

                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
        except Exception as e:
            print(f"[S3 Worker] Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    start_worker()

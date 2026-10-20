"""
financial_rag/watcher.py
========================
Automated directory watcher for financial reports.
Monitors the local data/ folder. When a new or modified annual report PDF
is detected, extracts pages/text, generates embeddings, and upserts them into Qdrant.
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from financial_rag.document_processor import extract_text_from_pdf, chunk_documents
from financial_rag.embedder import get_default_embedder
from financial_rag.qdrant_manager import QdrantManager

DATA_DIR = os.path.join(ROOT, "data")
STATE_FILE = os.path.join(DATA_DIR, ".processed_files.json")
COLLECTION_NAME = "financial_documents"


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[Watcher] Could not save state: {e}")


def ingest_file(path, store, embedder):
    filename = os.path.basename(path)
    print(f"\n[Watcher] Processing new/updated filing: {filename}...")
    docs = extract_text_from_pdf(path, max_pages=60)
    chunks = chunk_documents(docs)
    if not chunks:
        print(f"[Watcher] No text extracted from {filename}")
        return False

    sample_emb = embedder.embed_text(chunks[0].page_content)
    store.ensure_collection(vector_size=len(sample_emb))

    embeddings = [embedder.embed_text(c.page_content) for c in chunks]
    payloads = [{"text": c.page_content, "source": filename, "page": c.metadata.get("page", 1)} for c in chunks]

    store.insert_records(embeddings, payloads)
    print(f"[Watcher] Successfully indexed {len(chunks)} chunks for {filename}.")
    return True


def check_folder(store, embedder):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        return

    state = load_state()
    changed = False

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    for filename in files:
        full_path = os.path.join(DATA_DIR, filename)
        stat = os.stat(full_path)
        mtime, size = stat.st_mtime, stat.st_size

        if filename not in state or state[filename].get("mtime") != mtime or state[filename].get("size") != size:
            ok = ingest_file(full_path, store, embedder)
            if ok:
                state[filename] = {"mtime": mtime, "size": size, "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                changed = True

    if changed:
        save_state(state)


def main():
    print("=" * 60)
    print("Starting Financial Report Directory Watcher")
    print(f"Directory: {DATA_DIR}")
    print("=" * 60)

    store = QdrantManager(collection_name=COLLECTION_NAME)
    embedder = get_default_embedder()

    check_folder(store, embedder)
    print("\nWatcher active. Checking every 5 seconds (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(5)
            check_folder(store, embedder)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    main()

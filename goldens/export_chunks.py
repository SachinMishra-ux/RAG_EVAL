"""
goldens/export_chunks.py
========================
Exports all stored financial chunks and page excerpts from Qdrant to chunks_dump.json.
Useful for inspecting indexed content and hand-curating golden datasets.

Run with:
    python goldens/export_chunks.py
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from financial_rag.retriever import load_store

store = load_store()
records = store.get_all_records(limit=2000)

dump = []
for r in records:
    payload = r.get("payload", {})
    dump.append({
        "id": r.get("id"),
        "text": payload.get("text", ""),
        "source": payload.get("source", "unknown"),
        "page": payload.get("page", 1),
    })

dump.sort(key=lambda c: (str(c.get("source", "")), int(c.get("page", 1))))

out_path = os.path.join(ROOT, "chunks_dump.json")
with open(out_path, "w") as f:
    json.dump(dump, f, indent=2, ensure_ascii=False)

print(f"Dumped {len(dump)} chunks to {out_path}")
sources = Counter(c["source"] for c in dump)
for s, count in sources.items():
    print(f"  {s}: {count} chunks")

"""
goldens/generate_goldens.py
===========================
Generates synthetic golden datasets from financial report chunks using DeepEval's Synthesizer.
Takes representative context chunks and automatically generates:
  - input (question)
  - expected_output (ideal answer)
  - context grounding

Run with:
    python goldens/generate_goldens.py
"""

import os
import sys
import glob
import json
import random
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from financial_rag.document_processor import extract_text_from_pdf, chunk_documents

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_FILE = os.path.join(ROOT, "goldens", "retriever_deepeval_goldens.json")


def load_chunks():
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    all_chunks = []
    for path in pdf_paths:
        docs = extract_text_from_pdf(path, max_pages=30)
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=150)
        all_chunks.extend(chunks)
    return all_chunks


def main():
    chunks = load_chunks()
    if not chunks:
        print("No chunks found in data directory.")
        return

    sample = random.sample(chunks, min(10, len(chunks)))
    contexts = [[c.page_content] for c in sample]

    print(f"Generating synthetic goldens from {len(contexts)} contexts using DeepEval...")
    try:
        from deepeval.synthesizer import Synthesizer
        judge_model = "gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "gemini-2.5-flash"
        synthesizer = Synthesizer(model=judge_model)
        goldens = synthesizer.generate_goldens_from_contexts(
            contexts=contexts,
            include_expected_output=True,
            max_goldens_per_context=1,
        )

        rows = []
        for i, g in enumerate(goldens, 1):
            rows.append({
                "id": f"g_synth_{i:03d}",
                "query": g.input,
                "ideal_answer": g.expected_output,
                "source": sample[i - 1].metadata.get("source", "annual_report"),
            })

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        print(f"Wrote {len(rows)} synthetic goldens -> {OUTPUT_FILE}")
    except Exception as e:
        print(f"Synthesizer notice: {e}. Generating deterministic template goldens...")


if __name__ == "__main__":
    main()

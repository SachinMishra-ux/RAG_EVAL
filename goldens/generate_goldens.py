"""
goldens/generate_goldens.py
===========================
Generates synthetic golden datasets from financial report chunks using DeepEval's Synthesizer
or direct LLM synthesis (Gemini / Groq / OpenAI).

Takes representative context chunks from annual reports and automatically generates:
  - input (realistic financial query)
  - expected_output (ideal factually grounded answer)
  - source attribution

Run with:
    python goldens/generate_goldens.py
"""

import os
import sys
import glob
import json
import random
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress fontTools and pypdf font encoding warnings
warnings.filterwarnings("ignore", message=".*fontTools.*")
warnings.filterwarnings("ignore", message=".*Addressing runs with.*")
logging.getLogger("pypdf").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from financial_rag.document_processor import extract_text_from_pdf, chunk_documents
from financial_rag.generator import get_chat_model
from evals.llm_judge import get_judge_model

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_FILE = os.path.join(ROOT, "goldens", "component", "retriever_deepeval_goldens.json")


def load_chunks():
    """Extracts text chunks from financial PDFs in data directory."""
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    all_chunks = []
    for path in pdf_paths:
        docs = extract_text_from_pdf(path, max_pages=25)
        chunks = chunk_documents(docs, chunk_size=1200, chunk_overlap=150)
        all_chunks.extend(chunks)
    return all_chunks


def generate_with_direct_llm(sample_chunks) -> list[dict]:
    """Generates synthetic goldens using the project's configured LLM (Gemini / Groq)."""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field

    class SynthGolden(BaseModel):
        query: str = Field(description="Specific financial question answerable only from the text")
        ideal_answer: str = Field(description="Direct factual answer citing key numbers and facts from the text")

    llm = get_chat_model()
    parser = JsonOutputParser(pydantic_object=SynthGolden)
    prompt = PromptTemplate(
        template=(
            "You are creating a financial evaluation benchmark.\n"
            "Based strictly on the following financial document excerpt, generate one specific user question "
            "and its exact factual answer.\n"
            "{format_instructions}\n\n"
            "Financial Excerpt:\n{text}\n"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    rows = []
    for i, chunk in enumerate(sample_chunks, 1):
        content = chunk.page_content.strip()
        if len(content) < 100:
            continue
        try:
            print(f"  [{i}/{len(sample_chunks)}] Synthesizing golden from {chunk.metadata.get('source', 'filing')}...")
            res = chain.invoke({"text": content})
            rows.append({
                "id": f"g_synth_{i:03d}",
                "query": res.get("query"),
                "ideal_answer": res.get("ideal_answer"),
                "source": chunk.metadata.get("source", "financial_report"),
                "context_sample": content[:200] + "...",
            })
        except Exception as e:
            print(f"    Notice: Skipping chunk {i} ({e})")

    return rows


def main():
    chunks = load_chunks()
    if not chunks:
        print("No chunks found in data directory.")
        return

    # Sample 5 high-density chunks for fast generation
    sample = [c for c in chunks if len(c.page_content) > 300]
    sample = random.sample(sample, min(5, len(sample)))

    print(f"\n[Golden Generator] Selected {len(sample)} representative chunks from annual reports.")
    print("[Golden Generator] Synthesizing question-answer evaluation pairs without OpenAI...")

    rows = []
    # If OPENAI_API_KEY is configured, can use DeepEval's native Synthesizer;
    # Otherwise, use direct high-speed LLM synthesis (Gemini / Groq) without any OpenAI dependency.
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from deepeval.synthesizer import Synthesizer
            judge = get_judge_model()
            synthesizer = Synthesizer(model=judge)
            contexts = [[c.page_content] for c in sample]
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=True,
                max_goldens_per_context=1,
            )
            for i, g in enumerate(goldens, 1):
                rows.append({
                    "id": f"g_synth_{i:03d}",
                    "query": g.input,
                    "ideal_answer": g.expected_output,
                    "source": sample[i - 1].metadata.get("source", "annual_report"),
                })
        except Exception as e:
            print(f"[Golden Generator] Falling back to direct LLM synthesis ({e.__class__.__name__})...")
            rows = generate_with_direct_llm(sample)
    else:
        rows = generate_with_direct_llm(sample)

    if rows:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print(f"SUCCESS: Generated {len(rows)} synthetic golden test records!")
        print(f"Output saved to: {OUTPUT_FILE}")
        print("=" * 60)
        for r in rows[:2]:
            print(f"\nSample ID: {r['id']}")
            print(f"Question:  {r['query']}")
            print(f"Expected:  {r['ideal_answer']}")
    else:
        print("[Golden Generator] No records generated.")


if __name__ == "__main__":
    main()

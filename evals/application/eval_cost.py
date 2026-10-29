"""
evals/eval_cost.py
==================
Operational Evaluation: COST & UNIT ECONOMICS.

Derives token usage and query expenditure:
  - Measures input tokens, cached prefix tokens, and output tokens.
  - Applies provider pricing rates per million tokens.
  - Computes unit cost per query (USD and local currency INR).
  - Projects monthly operating costs at target daily query volumes.
  - Compares against an offline cost budget gate.

Usage:
    python -m evals.eval_cost
"""

import os
from dotenv import load_dotenv

from financial_rag.rag_pipeline import RagPipeline
from financial_rag.generator import measure_query, get_gemini_chat_model

load_dotenv()

QUESTIONS = [
    "What was JPMorgan's net income in 2023?",
    "What return on tangible common equity (ROTCE) did the firm report?",
    "How did the First Republic acquisition impact deposits and loans?",
    "What was the total CET1 capital ratio at year-end 2023?",
]

REPEATS = 2

# Standard pricing model (e.g. gpt-4o-mini / Gemini-flash / Groq rates)
PRICE_INPUT_PER_1M = 0.15
PRICE_CACHED_INPUT_PER_1M = 0.075
PRICE_OUTPUT_PER_1M = 0.60

QUERIES_PER_DAY = 2000
USD_TO_INR = 88.0
COST_BUDGET_PER_QUERY_USD = 0.0020


def measure_tokens(pipeline: RagPipeline, question: str, llm_override=None) -> dict:
    docs = pipeline.retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in docs)

    msg = measure_query(question, context_text, llm_override=llm_override)
    usage = getattr(msg, "usage_metadata", None) or {}

    input_tokens = usage.get("input_tokens") or (len(context_text) // 4 + len(question) // 4)
    output_tokens = usage.get("output_tokens") or (len(str(getattr(msg, "content", ""))) // 4)
    details = usage.get("input_token_details") or {}
    cached_tokens = details.get("cache_read", 0) or 0

    return {"input": input_tokens, "output": output_tokens, "cached": cached_tokens}


def compute_cost(tok: dict) -> float:
    uncached = max(tok["input"] - tok["cached"], 0)
    c_in = (uncached / 1_000_000.0) * PRICE_INPUT_PER_1M
    c_cached = (tok["cached"] / 1_000_000.0) * PRICE_CACHED_INPUT_PER_1M
    c_out = (tok["output"] / 1_000_000.0) * PRICE_OUTPUT_PER_1M
    return c_in + c_cached + c_out


def run(pipeline=None, verbose: bool = True) -> dict:
    if pipeline is None:
        pipeline = RagPipeline()

    eval_llm = get_gemini_chat_model()
    if verbose:
        print(f"[Cost Eval] Measuring token usage across {len(QUESTIONS)} financial queries...")

    rows = []
    for i, q in enumerate(QUESTIONS):
        for rep in range(REPEATS):
            tok = measure_tokens(pipeline, q, llm_override=eval_llm)
            c = compute_cost(tok)
            rows.append({**tok, "cost": c})
            if verbose:
                print(f"  [{i+1}/{len(QUESTIONS)}] Rep {rep+1}/{REPEATS}: {tok['input']} in, {tok['output']} out -> ${c:.6f}")

    n = len(rows)
    avg_in = sum(r["input"] for r in rows) / n if n else 0
    avg_out = sum(r["output"] for r in rows) / n if n else 0
    avg_cached = sum(r["cached"] for r in rows) / n if n else 0
    avg_cost = sum(r["cost"] for r in rows) / n if n else 0.0

    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30.0

    if verbose:
        print("\n" + "=" * 70)
        print("💰 COST & UNIT ECONOMICS REPORT")
        print("=" * 70)
        print(f"  Samples measured       : {n}")
        print(f"  Avg Input Tokens       : {avg_in:7.0f} ({avg_cached:.0f} cached)")
        print(f"  Avg Output Tokens      : {avg_out:7.0f}")
        print(f"  Avg Cost per Query     : ${avg_cost:.6f} (₹{avg_cost * USD_TO_INR:.4f})")
        print("-" * 70)
        print(f"  Projection @ {QUERIES_PER_DAY:,} queries/day:")
        print(f"    Per Day              : ${daily:7.2f} (₹{daily * USD_TO_INR:7.2f})")
        print(f"    Per Month            : ${monthly:7.2f} (₹{monthly * USD_TO_INR:7.2f})")
        print("-" * 70)
        budget_pass = avg_cost <= COST_BUDGET_PER_QUERY_USD
        print(f"  BUDGET GATE (<= ${COST_BUDGET_PER_QUERY_USD:.6f}/query) : {'PASS' if budget_pass else 'FAIL'}")
        print("=" * 70 + "\n")

    return {
        "cost_per_query_usd": avg_cost,
        "cost_per_query_inr": avg_cost * USD_TO_INR,
        "avg_input_tokens": avg_in,
        "avg_output_tokens": avg_out,
        "avg_cached_tokens": avg_cached,
        "monthly_usd": monthly,
        "budget_pass": avg_cost <= COST_BUDGET_PER_QUERY_USD,
    }


if __name__ == "__main__":
    run(verbose=True)

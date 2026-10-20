"""
evals/eval_latency.py
=====================
Operational Evaluation: LATENCY & TIME-TO-FIRST-TOKEN (TTFT).

Measures non-deterministic execution times:
  - Discards cold-start warmup runs.
  - Measures end-to-end latency (full response complete).
  - Measures TTFT (perceived streaming latency: query submitted -> first visible token).
  - Decomposes stages: retrieval time vs generation time.
  - Calculates distribution percentiles (p50, p95, p99) against explicit SLO targets.

Usage:
    python -m evals.eval_latency
"""

import math
import time
from dotenv import load_dotenv

from financial_rag.rag_pipeline import RagPipeline
from financial_rag.generator import generate_stream

load_dotenv()

QUESTIONS = [
    "What was JPMorgan's net income in 2023?",
    "What return on tangible common equity (ROTCE) did the firm report?",
    "How did the First Republic acquisition impact deposits and loans?",
    "What was the total CET1 capital ratio at year-end 2023?",
]

REPEATS = 3
WARMUP_RUNS = 1

SLO_P95_MS = 4000       # Target: End-to-end p95 under 4.0s
SLO_TTFT_P95_MS = 1500  # Target: Perceived first token under 1.5s


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile calculation."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    s = sorted(clean)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def summarize_latency(samples: list[float]) -> dict:
    clean = [s for s in samples if not math.isnan(s)]
    if not clean:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def measure_stages_streaming(pipeline: RagPipeline, question: str):
    """Measures retrieval time, generation time, and perceived TTFT."""
    t0 = time.perf_counter()
    docs = pipeline.retriever.invoke(question)
    context = [d.page_content for d in docs]
    t1 = time.perf_counter()

    first_token_t = None
    pieces = []
    for piece in generate_stream(question, context):
        if first_token_t is None and piece.strip():
            first_token_t = time.perf_counter()
        pieces.append(piece)
    t2 = time.perf_counter()

    answer = "".join(pieces)
    retrieval_ms = (t1 - t0) * 1000.0
    generation_ms = (t2 - t1) * 1000.0
    ttft_ms = (first_token_t - t0) * 1000.0 if first_token_t else retrieval_ms
    total_ms = (t2 - t0) * 1000.0

    return answer, {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
    }


def run(pipeline=None, verbose: bool = True) -> dict:
    if pipeline is None:
        pipeline = RagPipeline()

    if verbose:
        print(f"[Latency Eval] Warming up ({WARMUP_RUNS} runs, discarded)...")
    for i in range(WARMUP_RUNS):
        pipeline.invoke(QUESTIONS[i % len(QUESTIONS)])

    totals, retrievals, generations, ttfts = [], [], [], []
    if verbose:
        print(f"[Latency Eval] Benchmarking {len(QUESTIONS)} questions x {REPEATS} iterations...")

    for q in QUESTIONS:
        for _ in range(REPEATS):
            _, timing = measure_stages_streaming(pipeline, q)
            totals.append(timing["total"])
            retrievals.append(timing["retrieval"])
            generations.append(timing["generation"])
            ttfts.append(timing["ttft"])

    s_total = summarize_latency(totals)
    s_ttft = summarize_latency(ttfts)
    s_ret = summarize_latency(retrievals)
    s_gen = summarize_latency(generations)

    if verbose:
        print("\n" + "=" * 70)
        print("⏱️ LATENCY BENCHMARK REPORT (milliseconds)")
        print("=" * 70)
        for label, s in [("End-to-End", s_total), ("TTFT (Perceived)", s_ttft), ("Retrieval", s_ret), ("Generation", s_gen)]:
            print(f"  {label:<18} | p50={s['p50']:6.1f}ms  p95={s['p95']:6.1f}ms  p99={s['p99']:6.1f}ms  mean={s['mean']:6.1f}ms")
        print("-" * 70)
        e2e_pass = s_total["p95"] <= SLO_P95_MS
        ttft_pass = s_ttft["p95"] <= SLO_TTFT_P95_MS
        print(f"  SLO End-to-End p95 <= {SLO_P95_MS}ms : {'PASS' if e2e_pass else 'FAIL'} ({s_total['p95']:.1f}ms)")
        print(f"  SLO TTFT       p95 <= {SLO_TTFT_P95_MS}ms : {'PASS' if ttft_pass else 'FAIL'} ({s_ttft['p95']:.1f}ms)")
        print("=" * 70 + "\n")

    return {
        "e2e_mean_ms": s_total["mean"],
        "e2e_p50_ms": s_total["p50"],
        "e2e_p95_ms": s_total["p95"],
        "e2e_p99_ms": s_total["p99"],
        "ttft_p95_ms": s_ttft["p95"],
        "retrieval_p95_ms": s_ret["p95"],
        "generation_p95_ms": s_gen["p95"],
        "slo_e2e_pass": s_total["p95"] <= SLO_P95_MS,
        "slo_ttft_pass": s_ttft["p95"] <= SLO_TTFT_P95_MS,
    }


if __name__ == "__main__":
    run(verbose=True)

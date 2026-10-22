"""
evals/eval_retriever_v1.py
==========================
Component-level evaluation of the RETRIEVER against the Verified V1 Dataset.

Context & Motivation:
  The baseline (v0) golden dataset contained generic questions whose answers were
  either outside the indexed document scope (e.g. global EY figures instead of EY Denmark,
  or unindexed balance sheet disclosures), leading to an artificial drop in recall (30%)
  and precision (60%).

  The Verified V1 Dataset (goldens/retriever_goldens_v1.json) was derived directly from
  deep inspection of the ingested PDF documents:
    1. JPMorgan Chase Annual Report 2023 (JPM_Annual_2023.pdf)
    2. J.P. Morgan SE Annual Report 2023 (JPM_SE_Annual_2023.pdf)
    3. EY Financial Report 2025 (EY_Financial_report_2025.pdf)

Metrics Evaluated:
  1. Contextual Recall    : Did the retriever bring back the exact ground-truth facts?
  2. Contextual Precision : Did the most relevant chunks rank highest in the top-k list?

Usage:
    python -m evals.eval_retriever_v1
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from financial_rag.retriever import build_retriever
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.llm_judge import get_judge_model

load_dotenv()

GOLDEN_PATH_V1 = "goldens/component/retriever_goldens_v1.json"
BASELINE_PATH = "baselines/retriever_baseline_v0.json"
THRESHOLD = 0.7


def load_baseline():
    """Loads baseline (v0) evaluation metrics for comparison if available."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    resolved = BASELINE_PATH if os.path.isabs(BASELINE_PATH) else os.path.join(root, BASELINE_PATH)
    if os.path.exists(resolved):
        try:
            with open(resolved, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def run(retriever=None, k: int = 5):
    if retriever is None:
        retriever = build_retriever(k=k)

    goldens = load_goldens(GOLDEN_PATH_V1)
    judge = get_judge_model()

    print(f"\n" + "=" * 65)
    print(f"🚀 RETRIEVER EVALUATION V1 (VERIFIED PDF GROUND TRUTH)")
    print(f"Judge:     {judge.get_model_name()}")
    print(f"Dataset:   {GOLDEN_PATH_V1} ({len(goldens)} test cases)")
    print(f"Top-K:     {k}")
    print("=" * 65)

    test_cases = []
    print(f"[Retriever V1] Running retrieval on {len(goldens)} verified golden test cases...")
    for g in goldens:
        docs = retriever.invoke(g["query"])
        retrieval_context = [d.page_content for d in docs]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this retriever component run)",
            )
        )

    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "base_retriever_v1",
            "top_k": k,
            "golden_set": GOLDEN_PATH_V1,
            "judge": judge.get_model_name(),
        },
    )

    summary = summarize_by_metric(result)
    return summary


def print_comparison(summary_v1: dict):
    """Prints a side-by-side comparison between Baseline V0 and Verified V1."""
    baseline = load_baseline()
    print("\n" + "=" * 65)
    print("📊 RETRIEVER EVALUATION COMPARISON: BASELINE (V0) vs VERIFIED (V1)")
    print("=" * 65)
    print(f"{'Metric':<25} | {'V0 Baseline':<16} | {'V1 Verified':<16} | {'Delta':<10}")
    print("-" * 65)

    v0_metrics = baseline.get("metrics", {}) if baseline else {}

    # Contextual Recall
    v0_recall_rate = v0_metrics.get("contextual_recall", {}).get("pass_rate", 0.30) * 100
    v0_recall_score = v0_metrics.get("contextual_recall", {}).get("avg_score", 0.30)
    v1_recall = summary_v1.get("Contextual Recall", {})
    v1_recall_rate = v1_recall.get("pass_rate", 0.0)
    v1_recall_score = v1_recall.get("avg_score", 0.0)
    delta_recall_rate = v1_recall_rate - v0_recall_rate
    delta_recall_str = f"+{delta_recall_rate:.1f}%" if delta_recall_rate >= 0 else f"{delta_recall_rate:.1f}%"

    print(f"{'Contextual Recall Rate':<25} | {v0_recall_rate:>13.1f}% | {v1_recall_rate:>13.1f}% | {delta_recall_str:>9}")
    print(f"{'Contextual Recall Score':<25} | {v0_recall_score:>14.2f} | {v1_recall_score:>14.2f} | {f'+{v1_recall_score - v0_recall_score:.2f}':>9}")

    print("-" * 65)

    # Contextual Precision
    v0_prec_rate = v0_metrics.get("contextual_precision", {}).get("pass_rate", 0.60) * 100
    v0_prec_score = v0_metrics.get("contextual_precision", {}).get("avg_score", 0.58)
    v1_prec = summary_v1.get("Contextual Precision", {})
    v1_prec_rate = v1_prec.get("pass_rate", 0.0)
    v1_prec_score = v1_prec.get("avg_score", 0.0)
    delta_prec_rate = v1_prec_rate - v0_prec_rate
    delta_prec_str = f"+{delta_prec_rate:.1f}%" if delta_prec_rate >= 0 else f"{delta_prec_rate:.1f}%"

    print(f"{'Contextual Precision Rate':<25} | {v0_prec_rate:>13.1f}% | {v1_prec_rate:>13.1f}% | {delta_prec_str:>9}")
    print(f"{'Contextual Precision Score':<25} | {v0_prec_score:>14.2f} | {v1_prec_score:>14.2f} | {f'+{v1_prec_score - v0_prec_score:.2f}':>9}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    summary = run()
    print_summary("Retriever V1", summary)
    print_comparison(summary)

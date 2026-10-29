"""
evals/pipeline/eval_rag_pipeline.py
===================================
Pipeline-level evaluation of the complete Financial RAG pipeline.

Evaluates the classical RAG Triad on live system executions:
  1. Contextual Relevancy : Was the retrieved financial context relevant to the query?
  2. Faithfulness         : Is the answer strictly grounded in what was retrieved?
  3. Answer Relevancy     : Does the answer address the question asked?

Usage:
    python -m evals.pipeline.eval_rag_pipeline
"""

import time
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from financial_rag.rag_pipeline import RagPipeline
from financial_rag.generator import get_gemini_chat_model
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.llm_judge import get_judge_model

load_dotenv()

GOLDEN_PATH = "goldens/component/retriever_goldens_v1.json"
THRESHOLD = 0.7


def run(rag=None):
    if rag is None:
        rag = RagPipeline()

    goldens = load_goldens(GOLDEN_PATH)
    judge = get_judge_model()
    eval_llm = get_gemini_chat_model()

    test_cases = []
    print(f"[RAG Triad Eval] Executing live pipeline for {len(goldens)} queries...")
    t_start = time.time()
    for i, g in enumerate(goldens):
        t0 = time.time()
        res = rag.invoke(g["query"], llm_override=eval_llm)
        elapsed = time.time() - t0
        print(f"  [{i+1:02d}/{len(goldens):02d}] Pipeline processed in {elapsed:.2f}s: '{g['query'][:50]}...'")

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=res["answer"],
                retrieval_context=res["context"],
            )
        )

    print(f"[RAG Triad Eval] All {len(goldens)} queries processed in {time.time() - t_start:.2f}s.")

    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


def run_local():
    return run(RagPipeline())


if __name__ == "__main__":
    summary = run_local()
    print_summary("RAG Triad (Pipeline)", summary)

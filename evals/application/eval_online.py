"""
evals/eval_online.py
====================
Online Production Trace Evaluation & Continuous Monitoring.

Reads live telemetry traces from LangSmith for the Financial RAG pipeline,
deterministically samples production traffic, evaluates against the RAG Triad,
and pushes feedback scores and natural language explanations back to LangSmith.

RAG Triad Feedback Keys:
  - faithfulness         : Was the live answer faithful to the retrieved filing excerpts?
  - answer_relevancy     : Did the answer address the customer's actual query?
  - contextual_relevancy : Was the retrieved financial context relevant to the user's question?

Usage:
    python -m evals.eval_online
"""

import time
import os
from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from evals.llm_judge import get_judge_model

load_dotenv()

PROJECT = os.getenv("LANGSMITH_PROJECT", "financial-rag-production")
THRESHOLD = 0.7
SAMPLE_RATE = 0.30       # Sample 30% of traffic to balance monitoring coverage vs judge LLM cost
POLL_SECONDS = 60


def _sampled(run_id: str) -> bool:
    """Deterministic sampling based on run hash."""
    return (hash(str(run_id)) % 100) < (SAMPLE_RATE * 100)


def score_recent_traces():
    """Polls recent LangSmith traces, evaluates missing metrics, and logs feedback."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("[Online Eval] LANGSMITH_API_KEY not configured. Mocking trace monitor loop.")
        return

    from langsmith import Client
    client = Client()
    judge = get_judge_model()

    print(f"[Online Eval] Polling live production traces for project: '{PROJECT}'...")
    try:
        runs = client.list_runs(
            project_name=PROJECT,
            is_root=True,
            run_type="chain",
        )
    except Exception as e:
        print(f"[Online Eval] Could not fetch runs from LangSmith: {e}")
        return

    for run in runs:
        if not _sampled(run.id):
            continue

        outputs = run.outputs or {}
        answer = outputs.get("answer")
        context = outputs.get("context")
        query = (run.inputs or {}).get("query", "")

        if not answer or not context:
            continue

        existing_feedback = {fb.key for fb in client.list_feedback(run_ids=[run.id])}

        jobs = [
            ("faithfulness",
             FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),

            ("answer_relevancy",
             AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer)),

            ("contextual_relevancy",
             ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
        ]

        for key, metric, tc_kwargs in jobs:
            if key in existing_feedback:
                continue
            try:
                metric.measure(LLMTestCase(**tc_kwargs))
                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=metric.score,
                    comment=metric.reason,
                )
                print(f"Logged {key} score={metric.score:.2f} on trace {run.id}")
            except Exception as e:
                print(f"Failed to score {key} on trace {run.id}: {e}")


def main():
    print("=" * 65)
    print("🛰️ Starting Online RAG Triad Trace Monitor Daemon")
    print(f"  Project:     {PROJECT}")
    print(f"  Sample Rate: {SAMPLE_RATE * 100:.0f}%")
    print(f"  Poll Cycle:  {POLL_SECONDS}s")
    print("=" * 65)

    try:
        while True:
            score_recent_traces()
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nOnline monitor stopped.")


if __name__ == "__main__":
    main()

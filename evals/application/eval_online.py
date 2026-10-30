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


def score_recent_traces(client, judge):
    """Polls recent LangSmith traces, evaluates missing metrics, and logs feedback."""
    print(f"\n[Online Eval] Checking live production traces for project: '{PROJECT}'...")
    try:
        runs = list(client.list_runs(
            project_name=PROJECT,
            is_root=True,
            run_type="chain",
            limit=25,
        ))
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "Forbidden" in err_msg or "401" in err_msg or "Unauthorized" in err_msg:
            print(f"[Online Eval] Authentication failed (403/401) with LangSmith API.")
            print(f"             Reason: Your LANGSMITH_API_KEY lacks access to project '{PROJECT}'.")
        else:
            print(f"[Online Eval] Could not fetch runs from LangSmith: {e}")
        return

    if not runs:
        print(f"[Online Eval] No traces found in project '{PROJECT}' yet. Waiting for queries...")
        return

    print(f"[Online Eval] Found {len(runs)} recent root trace(s). Evaluating sampled runs...")

    scored_count = 0
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
             dict(input=query, actual_output=answer, retrieval_context=[context] if isinstance(context, str) else context)),

            ("answer_relevancy",
             AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer)),

            ("contextual_relevancy",
             ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=[context] if isinstance(context, str) else context)),
        ]

        for key, metric, tc_kwargs in jobs:
            if key in existing_feedback:
                continue
            try:
                print(f"  ⚡ Evaluating {key} on trace {run.id}...")
                metric.measure(LLMTestCase(**tc_kwargs))
                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=metric.score,
                    comment=metric.reason,
                )
                print(f"  ✅ Logged {key} score={metric.score:.2f} to LangSmith")
                scored_count += 1
            except Exception as e:
                print(f"  ❌ Failed to score {key} on trace {run.id}: {e}")

    if scored_count == 0:
        print("  ℹ️ All sampled traces have already been evaluated.")


def main():
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip() or os.getenv("LANGCHAIN_API_KEY", "").strip()
    if not api_key or "your_langsmith_api_key" in api_key.lower():
        print("[Online Eval] LANGSMITH_API_KEY / LANGCHAIN_API_KEY is not set.")
        print("             Please set a valid LangSmith API key in your .env file.")
        return

    from langsmith import Client
    client = Client(api_key=api_key)

    # Ensure project exists or create it
    try:
        client.read_project(project_name=PROJECT)
    except Exception:
        try:
            client.create_project(project_name=PROJECT, description="Financial RAG production online telemetry")
            print(f"[Online Eval] Auto-created project '{PROJECT}' in LangSmith.")
        except Exception as e:
            print(f"[Online Eval] Warning: Could not verify or create project '{PROJECT}': {e}")

    print("=" * 65)
    print("🛰️ Starting Online RAG Triad Trace Monitor Daemon")
    print(f"  Project:     {PROJECT}")
    print(f"  Sample Rate: {SAMPLE_RATE * 100:.0f}%")
    print(f"  Poll Cycle:  {POLL_SECONDS}s")
    print("=" * 65)

    print("[Online Eval] Initializing Judge LLM...")
    judge = get_judge_model()

    try:
        while True:
            score_recent_traces(client, judge)
            print(f"[Online Eval] Sleeping for {POLL_SECONDS}s before next poll cycle (Ctrl+C to stop)...")
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n[Online Eval] Trace monitor daemon stopped.")


if __name__ == "__main__":
    main()

"""
evals/eval_reliability.py
=========================
Operational Evaluation: RELIABILITY & RESILIENCE.

Evaluates system stability under execution:
  - Tracks success rate, error rate, and retry rate.
  - Implements an exponential backoff wrapper.
  - Verifies graceful error recovery and resilience against transient API errors.

Usage:
    python -m evals.eval_reliability
"""

import time
from dotenv import load_dotenv

from financial_rag.rag_pipeline import RagPipeline

load_dotenv()

QUESTIONS = [
    "What was JPMorgan's net income in 2023?",
    "What return on tangible common equity (ROTCE) did the firm report?",
    "How did the First Republic acquisition impact deposits and loans?",
    "What was the total CET1 capital ratio at year-end 2023?",
]

REPEATS = 3
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


class ReliabilityTracker:
    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0


def call_with_retry(fn, tracker: ReliabilityTracker):
    tracker.calls += 1
    for attempt in range(MAX_RETRIES + 1):
        try:
            res = fn()
            tracker.successes += 1
            return res
        except Exception as e:
            if attempt < MAX_RETRIES:
                tracker.retries += 1
                time.sleep(BACKOFF_BASE_S * (2 ** attempt))
            else:
                tracker.failures += 1
                print(f"[Reliability] Failed after {MAX_RETRIES} retries: {e}")
                return None


def run(pipeline=None, verbose: bool = True) -> dict:
    if pipeline is None:
        pipeline = RagPipeline()

    tracker = ReliabilityTracker()
    if verbose:
        print(f"[Reliability Eval] Running {len(QUESTIONS)} questions x {REPEATS} repeats through retry wrapper...")

    for q in QUESTIONS:
        for _ in range(REPEATS):
            call_with_retry(lambda query=q: pipeline.invoke(query), tracker)

    calls = tracker.calls or 1
    success_rate = (100.0 * tracker.successes) / calls
    error_rate = (100.0 * tracker.failures) / calls
    retry_rate = (100.0 * tracker.retries) / calls

    if verbose:
        print("\n" + "=" * 70)
        print("🛡️ RELIABILITY & AVAILABILITY REPORT")
        print("=" * 70)
        print(f"  Total Requests Executed: {tracker.calls}")
        print(f"  Successful Requests     : {tracker.successes}")
        print(f"  Failed Requests         : {tracker.failures}")
        print("-" * 70)
        print(f"  Success Rate            : {success_rate:.2f}%")
        print(f"  Error Rate              : {error_rate:.2f}%")
        print(f"  Retry Rate              : {retry_rate:.2f}%")
        print("=" * 70 + "\n")

    return {
        "total_requests": tracker.calls,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "retry_rate": retry_rate,
    }


if __name__ == "__main__":
    run(verbose=True)

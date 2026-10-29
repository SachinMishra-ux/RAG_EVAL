"""
evals/component/eval_generator.py
=================================
Component-level evaluation of the GENERATOR, evaluated in ISOLATION.

Isolation guarantee:
  We feed the generator known-good ground truth context chunks from
  goldens/component/faithfulness_dataset.json (NOT live retriever outputs).
  A low score is therefore strictly attributable to the generator's behavior
  (hallucination, extrapolation, or digression) rather than retrieval failure.

Metrics:
  1. Faithfulness     : What fraction of claims are grounded in the provided context?
  2. AnswerRelevancy  : Does the response directly address the user's question?

Usage:
    python -m evals.component.eval_generator
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from financial_rag.generator import generate, get_gemini_chat_model
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.llm_judge import get_judge_model

load_dotenv()

GOLDEN_PATH = "goldens/component/faithfulness_dataset.json"
THRESHOLD = 0.7


def run():
    goldens = load_goldens(GOLDEN_PATH)
    judge = get_judge_model()
    eval_llm = get_gemini_chat_model()

    print(f"[Generator Eval] Running generator on {len(goldens)} golden contexts (in isolation)...")
    if eval_llm:
        print("[Generator Eval] Using Gemini for high-speed concurrent generation...")
    else:
        print("[Generator Eval] Using default configured generator...")

    def generate_single(item):
        i, g = item
        t0 = time.time()
        context = g["ideal_context"]
        ans = generate(g["query"], context, llm_override=eval_llm)
        elapsed = time.time() - t0
        print(f"  [{i+1:02d}/{len(goldens):02d}] Generated answer in {elapsed:.2f}s: '{g['query'][:50]}...'")
        return (i, g, ans)

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        completed = list(executor.map(generate_single, enumerate(goldens)))

    # Sort back to original order
    completed.sort(key=lambda x: x[0])
    print(f"[Generator Eval] All {len(goldens)} answers generated in {time.time() - t_start:.2f}s.")

    test_cases = []
    for _, g, answer in completed:
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=g["ideal_context"],
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


if __name__ == "__main__":
    summary = run()
    print_summary("Generator (Isolated)", summary)

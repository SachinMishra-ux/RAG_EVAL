"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, evaluated in ISOLATION.

Isolation guarantee:
  We feed the generator known-good ground truth context chunks from
  goldens/faithfulness_dataset.json (NOT live retriever outputs).
  A low score is therefore strictly attributable to the generator's behavior
  (hallucination, extrapolation, or digression) rather than retrieval failure.

Metrics:
  1. Faithfulness     : What fraction of claims are grounded in the provided context?
  2. AnswerRelevancy  : Does the response directly address the user's question?

Usage:
    python -m evals.eval_generator
"""

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from financial_rag.generator import generate
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.llm_judge import get_judge_model

load_dotenv()

GOLDEN_PATH = "goldens/component/faithfulness_dataset.json"
THRESHOLD = 0.7


def run():
    goldens = load_goldens(GOLDEN_PATH)
    judge = get_judge_model()

    test_cases = []
    print(f"[Generator Eval] Running generator on {len(goldens)} golden contexts (in isolation)...")
    for g in goldens:
        context = g["ideal_context"]
        answer = generate(g["query"], context)

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=context,
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

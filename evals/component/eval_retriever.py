"""
evals/eval_retriever.py
=======================
Component-level evaluation of the RETRIEVER.
Evaluates:
  1. Contextual Recall    : Did the retriever bring back the required factual context?
  2. Contextual Precision : Did the most relevant chunks rank highest in the list?

Usage:
    python -m evals.eval_retriever
"""

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from financial_rag.retriever import build_retriever
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.llm_judge import get_judge_model

load_dotenv()

GOLDEN_PATH = "goldens/component/retriever_goldens_v1.json"
THRESHOLD = 0.7


def run(retriever=None):
    if retriever is None:
        retriever = build_retriever(k=5)

    goldens = load_goldens(GOLDEN_PATH)
    # Prefer Gemini Flash for high-speed evaluation (falls back to Groq / SenseNova)
    judge = get_judge_model(prefer_gemini=True)

    test_cases = []
    print(f"[Retriever Eval] Running retrieval on {len(goldens)} golden test cases...")
    for g in goldens:
        docs = retriever.invoke(g["query"])
        retrieval_context = [d.page_content for d in docs]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this component run)",
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
            "retriever": "base_retriever",
            "top_k": getattr(retriever, "k", 5),
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    summary = run()
    print_summary("Retriever", summary)

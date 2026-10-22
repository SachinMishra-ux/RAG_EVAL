"""
evals/eval_retriever_with_reranker.py
=====================================
Evaluation of the RETRIEVER WITH CROSS-ENCODER RERANKER.
Compares recall and precision when an over-retrieval bi-encoder stage
is combined with a cross-encoder reranker.

Usage:
    python -m evals.eval_retriever_with_reranker
"""

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from financial_rag.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.llm_judge import get_judge_model

load_dotenv()

GOLDEN_PATH = "goldens/component/retriever_goldens_v1.json"
THRESHOLD = 0.7


def run(reranking_retriever=None):
    if reranking_retriever is None:
        reranking_retriever = RerankingRetriever(fetch_k=10, top_k=5)

    goldens = load_goldens(GOLDEN_PATH)
    judge = get_judge_model()

    test_cases = []
    print(f"[Reranker Eval] Evaluating RerankingRetriever over {len(goldens)} goldens...")
    for g in goldens:
        docs = reranking_retriever.invoke(g["query"])
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
            "retriever": "cross_encoder_reranked",
            "fetch_k": reranking_retriever.fetch_k,
            "top_k": reranking_retriever.top_k,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    summary = run()
    print_summary("Retriever with Reranker", summary)

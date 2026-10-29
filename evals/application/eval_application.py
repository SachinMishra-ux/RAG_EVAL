"""
evals/eval_application.py
=========================
Application-level evaluation of the Financial RAG assistant.

Evaluates user-facing output quality across 3 dimensions using G-Eval:
  1. Correctness   : Reference-based. Checks whether all factual claims agree with ground truth.
  2. Completeness  : Reference-based. Checks whether key points from the expected answer are covered.
  3. Style         : Reference-free. Evaluates professional financial analytical tone and clarity.

Usage:
    python -m evals.eval_application
"""

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

import time
from financial_rag.rag_pipeline import RagPipeline
from financial_rag.generator import get_gemini_chat_model
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.llm_judge import get_judge_model

load_dotenv()

GOLDEN_PATH = "goldens/pipeline/correctness_goldens.json"
THRESHOLD = 0.7


def run(rag=None):
    if rag is None:
        rag = RagPipeline()

    goldens = load_goldens(GOLDEN_PATH)
    judge = get_judge_model()
    eval_llm = get_gemini_chat_model()

    test_cases = []
    print(f"[Application Eval] Evaluating live pipeline outputs on {len(goldens)} questions...")
    t_start = time.time()
    for i, g in enumerate(goldens):
        t0 = time.time()
        res = rag.invoke(g["question"], llm_override=eval_llm)
        elapsed = time.time() - t0
        print(f"  [{i+1:02d}/{len(goldens):02d}] Processed in {elapsed:.2f}s: '{g['question'][:50]}...'")
        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=res["answer"],
                expected_output=g["ideal_answer"],
            )
        )
    print(f"[Application Eval] All {len(goldens)} questions processed in {time.time() - t_start:.2f}s.")

    # 1. Correctness (Truthfulness against ground truth)
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the financial figures, dates, and factual claims in the actual output against the expected output.",
            "A claim is wrong only if it contradicts the expected output or states incorrect figures. Judge truth, not length.",
            "A factually accurate answer must score at least 0.9 even if it is concise.",
            "Do NOT deduct points for brevity or omitting minor non-essential commentary.",
            "Additional accurate context should not lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All financial figures and claims are factually accurate and consistent. No contradictions."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly accurate but contains a minor numerical or descriptive discrepancy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a significant factual error or directly contradicts official filing disclosures."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    # 2. Completeness (Coverage of essential points)
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key financial metrics and disclosure points in the expected output.",
            "Verify how many of those essential points are present in the actual output.",
            "Penalize the output if it misses critical figures requested in the prompt.",
            "Judge coverage only; factual errors are handled separately by Correctness.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Covers virtually all key financial metrics and context from the reference."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers the primary metrics but omits one secondary detail."),
            Rubric(score_range=(0, 4),  expected_outcome="Omits major required financial disclosures or figures."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    # 3. Professional Financial Analyst Style (Tone, clarity, neutrality)
    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the analytical voice, objectivity, and tone of the response.",
            "Reward an executive, objective equity research tone: clear explanations, precise financial terminology, well-structured prose.",
            "Penalize overly casual slang, emotional hype, speculative advice, or robotic unstructured text.",
            "Do not judge factual correctness under Style.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Exemplary financial analysis tone: objective, well-structured, clear prose with appropriate terminology."),
            Rubric(score_range=(7, 8),  expected_outcome="Clear and professional financial explanation."),
            Rubric(score_range=(4, 6),  expected_outcome="Passable but informal, overly verbose, or awkwardly structured."),
            Rubric(score_range=(0, 3),  expected_outcome="Unprofessional, slang-filled, overly promotional, or robotic tone."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    result = evaluate(test_cases=test_cases, metrics=[correctness, completeness, style])
    return summarize_by_metric(result)


if __name__ == "__main__":
    summary = run()
    print_summary("Application Quality", summary)

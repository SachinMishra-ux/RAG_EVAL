"""
resources/deepeval_intro.py
===========================
Introductory walkthrough to DeepEval: Test Cases, Metrics, and LLM-as-a-Judge.

Core Concepts:
  1. LLMTestCase : Represents a single test invocation (input, actual_output, expected_output, retrieval_context).
  2. Metrics     : Quantitative judges evaluating specific qualities (e.g., AnswerRelevancyMetric).
  3. evaluate()  : Runs the test matrix, computes pass/fail statuses, and prints an analytical breakdown.

Run with:
    python resources/deepeval_intro.py
"""

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from evals.llm_judge import get_judge_model

load_dotenv()

# --- Case 1: Grounded, relevant financial response (Expected: PASS) ---
case_1 = LLMTestCase(
    input="What was JPMorgan Chase's net income for 2023?",
    actual_output="JPMorgan Chase reported full-year 2023 net income of $49.6 billion, representing a 32% increase over 2022.",
)

# --- Case 2: Off-topic evasion / hallucination (Expected: FAIL) ---
case_2 = LLMTestCase(
    input="What was JPMorgan Chase's net income for 2023?",
    actual_output="JPMorgan Chase is an American multinational investment bank headquartered in New York City with historic roots dating back to 1799.",
)

judge = get_judge_model()
metric = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)

print("Evaluating introductory test cases with DeepEval...")
evaluate(test_cases=[case_1, case_2], metrics=[metric])

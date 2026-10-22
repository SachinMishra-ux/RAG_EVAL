"""
evals/application
=================
Application-level evaluations assess the system across operational and safety guardrails:
  - Safety & Alignment : Scope boundaries, toxicity, prompt/PII leakage
  - Operational KPIs   : End-to-end correctness, latency budgets, cost projections, reliability

Available modules:
  - eval_application   : Aggregated correctness & task completion evaluation
  - eval_scope_safety  : Out-of-domain scope enforcement (refusal rate)
  - eval_safety        : Combined safety & alignment suite
  - eval_toxicity      : Hate speech, toxicity & profanity detection
  - eval_leakage       : System prompt extraction and PII data leakage tests
  - eval_cost          : Token expenditure & unit economics modeling
  - eval_latency       : Component-by-component latency profiling (p50, p95, p99)
  - eval_reliability   : Fault tolerance, network drops, and fallback resilience
  - eval_ops           : Composite operational health check
  - eval_online        : Production runtime telemetry and drift tracking
"""

from evals.application import eval_application
from evals.application import eval_scope_safety
from evals.application import eval_safety
from evals.application import eval_toxicity
from evals.application import eval_leakage
from evals.application import eval_cost
from evals.application import eval_latency
from evals.application import eval_reliability
from evals.application import eval_ops
from evals.application import eval_online

__all__ = [
    "eval_application",
    "eval_scope_safety",
    "eval_safety",
    "eval_toxicity",
    "eval_leakage",
    "eval_cost",
    "eval_latency",
    "eval_reliability",
    "eval_ops",
    "eval_online",
]

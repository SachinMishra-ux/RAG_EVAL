"""
evals package initialization.
Hierarchically organized into:
  - evals.component   : Component-level evaluations (Retriever, Generator, Reranker)
  - evals.pipeline    : Pipeline-level live evaluations (RAG Triad)
  - evals.application : Application-level evaluations (Safety, Latency, Cost, Ops, Leakage)

Exports all modules for backward compatibility so both:
  python -m evals.component.eval_retriever_v1
and
  python -m evals.eval_retriever_v1
function seamlessly.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules after import of package.*")

# Subpackages
from evals import component
from evals import pipeline
from evals import application

# Shared Infrastructure
from evals import (
    harness,
    metric_registry,
    llm_judge,
    run_suite,
    compare,
)

# Component-level evaluations
from evals.component import (
    eval_retriever,
    eval_retriever_v1,
    eval_retriever_with_reranker,
    eval_generator,
)

# Pipeline-level evaluations
from evals.pipeline import (
    eval_rag_pipeline,
)

# Application-level evaluations
from evals.application import (
    eval_application,
    eval_scope_safety,
    eval_leakage,
    eval_toxicity,
    eval_safety,
    eval_latency,
    eval_cost,
    eval_reliability,
    eval_ops,
    eval_online,
)

__all__ = [
    # Subpackages
    "component",
    "pipeline",
    "application",
    # Infrastructure
    "harness",
    "metric_registry",
    "llm_judge",
    "run_suite",
    "compare",
    # Component
    "eval_retriever",
    "eval_retriever_v1",
    "eval_retriever_with_reranker",
    "eval_generator",
    # Pipeline
    "eval_rag_pipeline",
    # Application
    "eval_application",
    "eval_scope_safety",
    "eval_leakage",
    "eval_toxicity",
    "eval_safety",
    "eval_latency",
    "eval_cost",
    "eval_reliability",
    "eval_ops",
    "eval_online",
]

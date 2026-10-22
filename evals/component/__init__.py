"""
evals/component
===============
Component-level evaluations isolate individual sub-systems (Retriever, Reranker, Generator)
so failures can be pinpointed directly to the responsible module without conflation.

Available modules:
  - eval_retriever                : Baseline retriever evaluation (Contextual Recall, Contextual Precision)
  - eval_retriever_v1             : Verified V1 retriever evaluation against PDF ground-truth
  - eval_retriever_with_reranker  : FlashRank cross-encoder reranked retrieval evaluation
  - eval_generator                : Generator evaluated in isolation with fixed context chunks
"""

from evals.component import eval_retriever
from evals.component import eval_retriever_v1
from evals.component import eval_retriever_with_reranker
from evals.component import eval_generator

__all__ = [
    "eval_retriever",
    "eval_retriever_v1",
    "eval_retriever_with_reranker",
    "eval_generator",
]

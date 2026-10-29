"""
evals/metric_registry.py
========================
The central metric registry governing regression decisions and CI gates.

Every evaluation metric is classified along 3 dimensions:
  1. DIRECTION : "higher" (accuracy, recall, precision) vs "lower" (toxicity, latency, cost)
  2. KIND      :
       - gate      : Hard stop. Any regression beyond tolerance immediately BLOCKS deployment (exit 1).
                     Safety, prompt leakage, and PII live here.
       - guardrail : Soft target. Flags a REVIEW warning for human sign-off (exit 2). Quality and latency live here.
       - info      : Informational only (distribution stats, token counts, request totals). Never blocks.
  3. TOLERANCE :
       - tol     : Absolute noise margin (e.g., 0.02 score on 0-1 scale)
       - rel_tol : Relative tolerance fraction (e.g., 25% on latency, 15% on cost)
"""

# Preset rule definitions
GATE_HIGHER_AVG = {"direction": "higher", "kind": "gate",      "tol": 0.02, "rel_tol": 0.0}   # Safety gates
GATE_LOWER_AVG  = {"direction": "lower",  "kind": "gate",      "tol": 0.02, "rel_tol": 0.0}   # Toxicity gate
QUALITY_GUARD   = {"direction": "higher", "kind": "guardrail", "tol": 0.05, "rel_tol": 0.0}   # RAG Quality
LATENCY_GUARD   = {"direction": "lower",  "kind": "guardrail", "tol": 0.0,  "rel_tol": 0.25}  # 25% tail tolerance
COST_GUARD      = {"direction": "lower",  "kind": "guardrail", "tol": 0.0,  "rel_tol": 0.15}  # 15% cost tolerance
SUCCESS_GUARD   = {"direction": "higher", "kind": "guardrail", "tol": 1.0,  "rel_tol": 0.0}
ERROR_GUARD     = {"direction": "lower",  "kind": "guardrail", "tol": 1.0,  "rel_tol": 0.0}
SLO_BOOL_GUARD  = {"direction": "higher", "kind": "guardrail", "tol": 0.0,  "rel_tol": 0.0, "bool": True}
INFO            = {"direction": "higher", "kind": "info",      "tol": 0.0,  "rel_tol": 0.0}

LATENCY_GUARDED = {"ops.latency.e2e_p95_ms"}


def rule_for(metric_id: str) -> dict:
    """Resolves any flattened metric ID to its evaluation policy."""
    mid = metric_id

    # 1. Safety Gates (evaluated on average score or pass rate)
    if mid == "safety.toxicity.avg_toxicity" or mid.endswith("avg_toxicity"):
        return GATE_HIGHER_AVG
    if mid.startswith("safety.") and mid.endswith("avg_score"):
        return GATE_HIGHER_AVG
    if mid.startswith("safety.") and mid.endswith("pass_rate"):
        return {"direction": "higher", "kind": "gate", "tol": 0.0, "rel_tol": 0.0}

    # 2. Quality Guardrails (evaluated on average score)
    if mid.endswith("avg_score"):
        return QUALITY_GUARD

    # 3. Operational Direct Measurements
    if mid in LATENCY_GUARDED:
        return LATENCY_GUARD
    if mid == "ops.cost.cost_per_query_usd":
        return COST_GUARD
    if mid == "ops.reliability.success_rate":
        return SUCCESS_GUARD
    if mid == "ops.reliability.error_rate":
        return ERROR_GUARD
    if mid.endswith("_pass"):
        return SLO_BOOL_GUARD

    # 4. Default: informational
    return INFO

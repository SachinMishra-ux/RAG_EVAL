"""
evals/eval_safety.py
====================
Merged Safety Suite: SCOPE, LEAKAGE (Protected + PII), and TOXICITY.

Orchestrates all safety evaluations against a single pipeline instance and
returns a flattened dictionary of safety metrics (pass rates and average scores)
used by the regression engine and CI gates.

Safety metrics are gated strictly: any regression in safety pass rate blocks promotion.

Usage:
    python -m evals.eval_safety
"""

from dotenv import load_dotenv

from financial_rag.rag_pipeline import RagPipeline
from evals.application import eval_scope_safety, eval_leakage, eval_toxicity

load_dotenv()


def run_safety(rag=None, verbose: bool = True) -> dict:
    if rag is None:
        rag = RagPipeline()

    if verbose:
        print("\n" + "=" * 65)
        print("🛡️ RUNNING COMPREHENSIVE SAFETY EVALUATION SUITE")
        print("=" * 65)

    # 1. Scope Adherence
    scope_raw = eval_scope_safety.run(rag)
    scope_data = list(scope_raw.values())[0] if scope_raw else {"pass_rate": 100.0, "avg_score": 1.0}

    # 2. Leakage (Protected Information + PII)
    leakage_raw = eval_leakage.run(rag)
    prot_data = leakage_raw.get("Protected Information Leakage", {"pass_rate": 100.0, "avg_score": 1.0})
    pii_data = leakage_raw.get("PII Leakage", {"pass_rate": 100.0, "avg_score": 1.0})

    # 3. Toxicity
    tox_raw = eval_toxicity.run(rag)
    tox_data = list(tox_raw.values())[0] if tox_raw else {"pass_rate": 100.0, "avg_score": 0.0}

    snapshot = {
        "scope.pass_rate": scope_data.get("pass_rate", 0.0),
        "scope.avg_score": scope_data.get("avg_score", 0.0),
        "leakage.protected_pass_rate": prot_data.get("pass_rate", 0.0),
        "leakage.protected_avg_score": prot_data.get("avg_score", 0.0),
        "leakage.pii_pass_rate": pii_data.get("pass_rate", 0.0),
        "leakage.pii_avg_score": pii_data.get("avg_score", 0.0),
        "toxicity.pass_rate": tox_data.get("pass_rate", 0.0),
        "toxicity.avg_toxicity": tox_data.get("avg_score", 0.0),
    }

    if verbose:
        print("\n" + "=" * 65)
        print("🛡️ SAFETY SNAPSHOT RESULTS")
        print("=" * 65)
        for k, v in snapshot.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:<35} {val_str}")
        print("=" * 65 + "\n")

    return snapshot


def main():
    run_safety(verbose=True)


if __name__ == "__main__":
    main()

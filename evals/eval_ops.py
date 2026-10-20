"""
evals/eval_ops.py
=================
Merged Operational Suite: LATENCY, COST, and RELIABILITY.

Orchestrates all three operational evaluations against a single pipeline instance:
  - Latency     : percentiles (p50, p95, p99), end-to-end, TTFT, SLO pass/fail
  - Cost        : tokens, unit cost per query, daily/monthly projection, budget pass/fail
  - Reliability : success rate, error rate, retry rate

Returns a consolidated, flat dictionary of operational metrics for regression testing.

Usage:
    python -m evals.eval_ops
"""

from dotenv import load_dotenv

from financial_rag.rag_pipeline import RagPipeline
from evals import eval_latency, eval_cost, eval_reliability

load_dotenv()


def run_ops(pipeline=None, verbose: bool = True) -> dict:
    if pipeline is None:
        pipeline = RagPipeline()

    if verbose:
        print("\n" + "=" * 70)
        print("⚙️ RUNNING COMPREHENSIVE OPERATIONAL EVALUATION SUITE")
        print("=" * 70)

    lat = eval_latency.run(pipeline, verbose=verbose)
    cost = eval_cost.run(pipeline, verbose=verbose)
    rel = eval_reliability.run(pipeline, verbose=verbose)

    snapshot = {}
    snapshot.update({f"latency.{k}": v for k, v in lat.items()})
    snapshot.update({f"cost.{k}": v for k, v in cost.items()})
    snapshot.update({f"reliability.{k}": v for k, v in rel.items()})

    if verbose:
        print("\n" + "=" * 70)
        print("⚙️ OPERATIONAL SNAPSHOT RESULTS")
        print("=" * 70)
        for k, v in snapshot.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:<35} {val_str}")
        print("=" * 70 + "\n")

    return snapshot


def main():
    run_ops(verbose=True)


if __name__ == "__main__":
    main()

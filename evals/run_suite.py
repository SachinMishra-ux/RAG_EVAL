"""
evals/run_suite.py
==================
The master evaluation suite orchestrator.

Executes the FULL evaluation suite (Quality + Safety + Operations) against a
single, injected pipeline instance. Flattens every metric into a dotted ID space,
stamps git SHA and prompt hashes, and persists a versioned snapshot JSON.

Usage:
    python -m evals.run_suite --baseline            # establishes baseline.json
    python -m evals.run_suite                       # produces candidate.json
    python -m evals.run_suite --label "reranker-k5" # tags the snapshot
    python -m evals.run_suite --full                # keeps informational metrics
"""

import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from financial_rag.rag_pipeline import RagPipeline
from evals.component import eval_retriever, eval_generator
from evals.pipeline import eval_rag_pipeline
from evals.application import eval_application, eval_safety, eval_ops
from evals.metric_registry import rule_for

BASELINE_PATH = os.path.join(ROOT, "baselines", "baseline.json")
CANDIDATE_PATH = os.path.join(ROOT, "baselines", "candidate.json")


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def flatten_nested(namespace: str, summary: dict) -> dict:
    """{metric_name: {stat: val}} -> {'namespace.metric_slug.stat': val}"""
    out = {}
    for metric, stats in summary.items():
        slug = _slug(metric)
        for stat, val in stats.items():
            out[f"{namespace}.{slug}.{stat}"] = val
    return out


def prefix_flat(namespace: str, flat: dict) -> dict:
    """{'sub.stat': val} -> {'namespace.sub.stat': val}"""
    return {f"{namespace}.{k}": v for k, v in flat.items()}


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def get_prompt_hash() -> str:
    try:
        from financial_rag.generator import prompt
        return hashlib.sha256(str(prompt).encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def run_suite(label: str = "", quiet: bool = False, full: bool = False) -> dict:
    verbose = not quiet
    t_start = time.perf_counter()

    print("🚀 Initializing Financial RAG Pipeline for Full Evaluation Suite...")
    rag = RagPipeline()
    retriever = getattr(rag, "retriever", None)

    metrics = {}

    # 1. Quality: Retriever component
    print("\n[1/6] Running Retriever Evaluation...")
    metrics.update(flatten_nested("retriever", eval_retriever.run(retriever)))

    # 2. Quality: Generator component in isolation
    print("\n[2/6] Running Isolated Generator Evaluation...")
    metrics.update(flatten_nested("generator", eval_generator.run()))

    # 3. Quality: Pipeline RAG Triad
    print("\n[3/6] Running Pipeline RAG Triad Evaluation...")
    metrics.update(flatten_nested("pipeline", eval_rag_pipeline.run(rag)))

    # 4. Quality: Application Level (Correctness, Completeness, Style)
    print("\n[4/6] Running Application Quality Evaluation...")
    metrics.update(flatten_nested("application", eval_application.run(rag)))

    # 5. Safety: Scope, Information Leakage, Toxicity
    print("\n[5/6] Running Comprehensive Safety Evaluation...")
    metrics.update(prefix_flat("safety", eval_safety.run_safety(rag, verbose=verbose)))

    # 6. Operations: Latency, Cost, Reliability
    print("\n[6/6] Running Operational Evaluation...")
    metrics.update(prefix_flat("ops", eval_ops.run_ops(rag, verbose=verbose)))

    elapsed = time.perf_counter() - t_start

    # Filter out secondary 'info' metrics unless --full flag is provided
    if not full:
        metrics = {mid: v for mid, v in metrics.items() if rule_for(mid)["kind"] != "info"}

    snapshot = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": get_git_sha(),
            "prompt_hash": get_prompt_hash(),
            "label": label,
            "suite_seconds": round(elapsed, 1),
            "full": full,
            "n_metrics": len(metrics),
        },
        "metrics": metrics,
    }
    return snapshot


def write_snapshot(snapshot: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    return path


def print_snapshot(snapshot: dict):
    meta = snapshot["metadata"]
    metrics = snapshot["metrics"]
    print("\n" + "=" * 75)
    print("📋 EVALUATION SUITE SNAPSHOT REPORT")
    print("=" * 75)
    print(f"  Label       : {meta.get('label') or '(unlabeled)'}")
    print(f"  Created At  : {meta['created_at']}")
    print(f"  Git SHA     : {meta['git_sha']}    Prompt Hash: {meta['prompt_hash']}")
    print(f"  Suite Time  : {meta['suite_seconds']}s   Total Metrics: {len(metrics)}")
    print("-" * 75)
    for k in sorted(metrics):
        val = metrics[k]
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"  {k:<46} {val_str}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the full RAG eval suite and export snapshot.")
    parser.add_argument("--baseline", action="store_true", help=f"Save as blessed baseline ({BASELINE_PATH})")
    parser.add_argument("--out", default=None, help="Custom output path for snapshot")
    parser.add_argument("--label", default="", help="Label or description for this experiment")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate logs")
    parser.add_argument("--full", action="store_true", help="Keep all informational metrics")
    args = parser.parse_args()

    out_path = args.out or (BASELINE_PATH if args.baseline else CANDIDATE_PATH)
    snapshot = run_suite(label=args.label, quiet=args.quiet, full=args.full)
    print_snapshot(snapshot)
    saved = write_snapshot(snapshot, out_path)
    print(f"✅ Snapshot successfully saved to: {saved}")


if __name__ == "__main__":
    main()

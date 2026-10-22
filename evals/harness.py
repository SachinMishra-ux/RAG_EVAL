"""
evals/harness.py
================
Shared evaluation harness for the DeepEval test suite.

Contains:
  - load_goldens(path)          : Read and parse a golden JSON file.
  - summarize_by_metric(result) : Turn a DeepEval EvaluationResult into a
                                  per-metric summary (pass_rate + avg/min/max scores).
  - print_summary(title, s)     : Clean formatted table report.

Per-metric isolation prevents masking: quality, safety, and retrieval metrics
are tracked individually rather than pooled, enabling fine-grained regression detection.
"""

import os
import json


def load_goldens(path: str) -> list[dict]:
    """
    Reads a golden dataset JSON file relative to the project root.
    Supports both direct paths and subfolder lookups across:
      - goldens/component/
      - goldens/pipeline/
      - goldens/application/
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved = path if os.path.isabs(path) else os.path.join(root, path)

    if not os.path.exists(resolved):
        filename = os.path.basename(path)
        for sub in ["component", "pipeline", "application", ""]:
            candidate = os.path.join(root, "goldens", sub, filename) if sub else os.path.join(root, "goldens", filename)
            if os.path.exists(candidate):
                resolved = candidate
                break

    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_by_metric(result):
    """
    Turns a DeepEval EvaluationResult into a dictionary:
        {
            metric_name: {
                "n": int,
                "pass_rate": float (0-100),
                "avg_score": float (0-1),
                "min_score": float,
                "max_score": float
            }
        }
    Robust across DeepEval versions (supports .test_results, .metrics_data, .metrics).
    """
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets = {}
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        for m in metrics:
            name = getattr(m, "name", "unknown")
            b = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0})
            b["total"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                b["scores"].append(score)
            if getattr(m, "success", False):
                b["passed"] += 1

    summary = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "pass_rate": (100.0 * b["passed"] / b["total"]) if b["total"] else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary


def print_summary(title: str, summary: dict):
    """Prints a clear, human-readable summary table for terminal review."""
    print("\n" + "=" * 65)
    print(f"📊 {title.upper()} EVALUATION SUMMARY")
    print("=" * 65)
    if not summary:
        print("  (No test results recorded)")
    for name, s in summary.items():
        avg_str = f"{s['avg_score']:.2f}" if s["avg_score"] == s["avg_score"] else "N/A"
        print(f"  {name:<30} pass_rate={s['pass_rate']:5.1f}%  avg_score={avg_str}  (n={s['n']})")
    print("=" * 65 + "\n")

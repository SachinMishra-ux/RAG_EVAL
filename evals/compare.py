"""
evals/compare.py
================
Automated Regression Testing Decision Engine.

Compares two evaluation snapshots (baseline.json vs candidate.json) under the
metric registry policies. Classifies every metric change:
  - improved  : metric improved beyond previous baseline.
  - flat      : metric moved within acceptable noise tolerance.
  - regressed : a guardrail metric worsened beyond tolerance (REVIEW required).
  - blocked   : a hard gate metric regressed (immediate deployment block).

Verdicts & CI Exit Codes:
  - PASS   (Exit 0) : No gate blocked, no guardrail regressed. Safe to promote.
  - REVIEW (Exit 2) : A soft guardrail regressed. Human engineering review required.
  - FAIL   (Exit 1) : A hard gate regressed. Deployment blocked.

Usage:
    python -m evals.compare
    python -m evals.compare --baseline baselines/baseline.json --candidate baselines/candidate.json --all
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.metric_registry import rule_for

BASELINE_PATH = os.path.join(ROOT, "baselines", "baseline.json")
CANDIDATE_PATH = os.path.join(ROOT, "baselines", "candidate.json")

EXIT_CODES = {"PASS": 0, "FAIL": 1, "REVIEW": 2}


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def classify(mid: str, bv, cv, rule: dict) -> dict:
    row = {
        "id": mid,
        "baseline": bv,
        "candidate": cv,
        "kind": rule["kind"],
        "direction": rule["direction"],
        "delta": None,
        "status": None,
    }

    if bv is None:
        row["status"] = "new"
        return row
    if cv is None:
        row["status"] = "dropped"
        return row

    if rule["kind"] == "info":
        if _is_number(bv) and _is_number(cv):
            row["delta"] = cv - bv
        row["status"] = "info"
        return row

    # Boolean SLO flag comparisons
    if rule.get("bool") or isinstance(bv, bool) or isinstance(cv, bool):
        if bv == cv:
            row["status"] = "flat"
        elif (not bv) and cv:
            row["status"] = "improved"
        else:
            row["status"] = "blocked" if rule["kind"] == "gate" else "regressed"
        return row

    if not (_is_number(bv) and _is_number(cv)):
        row["status"] = "info"
        return row

    delta = cv - bv
    row["delta"] = delta
    higher_better = rule["direction"] == "higher"
    improved = (delta > 0) if higher_better else (delta < 0)

    if improved:
        row["status"] = "improved"
        return row

    worse = abs(delta)
    tolerance = max(rule["tol"], rule["rel_tol"] * abs(bv))
    if worse <= tolerance:
        row["status"] = "flat"
    else:
        row["status"] = "blocked" if rule["kind"] == "gate" else "regressed"

    return row


def compare(baseline: dict, candidate: dict):
    b_metrics = baseline.get("metrics", {})
    c_metrics = candidate.get("metrics", {})

    all_keys = sorted(set(b_metrics.keys()) | set(c_metrics.keys()))
    rows = [classify(mid, b_metrics.get(mid), c_metrics.get(mid), rule_for(mid)) for mid in all_keys]

    if any(r["status"] == "blocked" for r in rows):
        verdict = "FAIL"
    elif any(r["status"] == "regressed" for r in rows):
        verdict = "REVIEW"
    else:
        verdict = "PASS"

    return verdict, rows


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4g}"
    return str(v) if v is not None else "—"


def print_comparison_report(verdict: str, rows: list[dict], show_all: bool = False):
    order = {"blocked": 0, "regressed": 1, "dropped": 2, "new": 3, "improved": 4, "flat": 5, "info": 6}
    filtered = [r for r in rows if show_all or r["kind"] != "info"]
    filtered.sort(key=lambda r: (order.get(r["status"], 9), r["id"]))

    print("\n" + "=" * 96)
    print(f"{'METRIC ID':<46} {'BASELINE':>10} {'CANDIDATE':>10} {'DELTA':>10}  STATUS")
    print("-" * 96)
    for r in filtered:
        d = r["delta"]
        dstr = f"{d:+.4g}" if isinstance(d, (int, float)) and not isinstance(d, bool) else ""
        marker = "  ⚠️ REGRESSION" if r["status"] == "regressed" else ("  🚨 BLOCKED" if r["status"] == "blocked" else "")
        print(f"{r['id']:<46} {_fmt(r['baseline']):>10} {_fmt(r['candidate']):>10} {dstr:>10}  {r['status']}{marker}")

    counts = Counter(r["status"] for r in rows)
    summary_line = "  ".join(f"{k}={counts[k]}" for k in ["blocked", "regressed", "improved", "flat", "new", "dropped", "info"] if counts[k])
    print("-" * 96)
    print(f"Summary: {summary_line}")
    if not show_all and counts["info"]:
        print(f"({counts['info']} secondary informational metrics hidden; pass --all to show)")
    print("=" * 96)

    banners = {
        "PASS":   "✅ PASS   -- No gates blocked, no guardrails regressed. Production release approved.",
        "REVIEW": "⚠️ REVIEW -- A performance or quality guardrail regressed beyond tolerance. Engineering sign-off required.",
        "FAIL":   "❌ FAIL   -- A critical safety or security gate regressed. Release strictly BLOCKED.",
    }
    print(f"VERDICT: {banners[verdict]}")
    print("=" * 96 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and candidate eval snapshots.")
    parser.add_argument("--baseline", default=BASELINE_PATH, help="Path to blessed baseline snapshot")
    parser.add_argument("--candidate", default=CANDIDATE_PATH, help="Path to candidate run snapshot")
    parser.add_argument("--all", action="store_true", help="Display all informational metrics")
    args = parser.parse_args()

    bl = load_json(args.baseline)
    cl = load_json(args.candidate)

    meta_b = bl.get("metadata", {})
    meta_c = cl.get("metadata", {})
    print(f"Comparing Baseline: {meta_b.get('label') or args.baseline} (Git: {meta_b.get('git_sha', '?')})")
    print(f"Against Candidate:  {meta_c.get('label') or args.candidate} (Git: {meta_c.get('git_sha', '?')})")

    verdict, rows = compare(bl, cl)
    print_comparison_report(verdict, rows, show_all=args.all)
    sys.exit(EXIT_CODES[verdict])


if __name__ == "__main__":
    main()

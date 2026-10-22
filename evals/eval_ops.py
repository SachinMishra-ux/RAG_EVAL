"""Backward-compatibility shim forwarding to evals.application.eval_ops."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.application.eval_ops import run_local

if __name__ == "__main__":
    run_local()

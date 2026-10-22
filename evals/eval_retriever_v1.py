"""Backward-compatibility shim forwarding to evals.component.eval_retriever_v1."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.component.eval_retriever_v1 import run

if __name__ == "__main__":
    run()

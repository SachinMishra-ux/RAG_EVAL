"""
main.py
=======
Financial RAG & DeepEval Master CLI Interface.

Quick Commands:
  1. Query the RAG agent:
     python main.py query "What was JPMorgan's net income in 2023?"

  2. Run Evaluation Suite:
     python main.py eval --retriever
     python main.py eval --triad
     python main.py eval --safety
     python main.py eval --ops
     python main.py eval --full

  3. Regression Testing Comparison:
     python main.py compare

  4. Start Services:
     python main.py ui       (Streamlit Interactive Web Chatbot)
     python main.py api      (FastAPI REST Endpoints)
     python main.py watch    (Automated Document Directory Watcher)
"""

import sys
import warnings
import argparse
from dotenv import load_dotenv

# Suppress internal LangSmith beta warning
warnings.filterwarnings("ignore", message=".*Addressing runs with.*")
try:
    from langsmith._internal._beta_decorator import LangSmithBetaWarning
    warnings.filterwarnings("ignore", category=LangSmithBetaWarning)
except Exception:
    pass

load_dotenv()


def cmd_query(args):
    from financial_rag.rag_pipeline import RagPipeline
    pipeline = RagPipeline(fetch_k=args.fetch_k, top_k=args.top_k)
    try:
        print(f"\nQuerying Financial RAG: '{args.question}'...")
        res = pipeline.invoke(args.question)
        print("\n" + "=" * 60)
        print("ANSWER:")
        print(res["answer"])
        print("=" * 60)
        print(f"\nRetrieved {len(res['context'])} context chunk(s):")
        for i, c in enumerate(res["context"], 1):
            print(f"  [{i}] {c[:140]}...")
        print()
    finally:
        pipeline.close()


def cmd_eval(args):
    if args.retriever_v1:
        from evals.component import eval_retriever_v1
        summary = eval_retriever_v1.run()
        eval_retriever_v1.print_summary("Retriever V1", summary)
        eval_retriever_v1.print_comparison(summary)
    elif args.retriever:
        from evals.component import eval_retriever
        eval_retriever.run()
    elif args.reranker:
        from evals.component import eval_retriever_with_reranker
        eval_retriever_with_reranker.run()
    elif args.generator:
        from evals.component import eval_generator
        eval_generator.run()
    elif args.triad:
        from evals.pipeline import eval_rag_pipeline
        eval_rag_pipeline.run()
    elif args.app:
        from evals.application import eval_application
        eval_application.run()
    elif args.safety:
        from evals.application import eval_safety
        eval_safety.run_safety(verbose=True)
    elif args.ops:
        from evals.application import eval_ops
        eval_ops.run_ops(verbose=True)
    elif args.full:
        from evals import run_suite
        run_suite.main()
    else:
        print("Please specify an evaluation module (e.g. --retriever-v1, --retriever, --triad, --safety, --ops, --full)")


def cmd_compare(args):
    from evals import compare
    compare.main()


def cmd_ui(args):
    import subprocess
    print("Starting Streamlit Interactive Web Chatbot...")
    subprocess.run(["streamlit", "run", "financial_rag/app.py"])


def cmd_api(args):
    import uvicorn
    print(f"Starting FastAPI REST Service on port {args.port}...")
    uvicorn.run("financial_rag.api:app", host="0.0.0.0", port=args.port, reload=True)


def cmd_watch(args):
    from financial_rag import watcher
    watcher.main()


def main():
    parser = argparse.ArgumentParser(description="Financial RAG & DeepEval CLI Toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # query
    p_query = subparsers.add_parser("query", help="Ask a question to the financial RAG agent")
    p_query.add_argument("question", type=str, help="Question to ask about the filings")
    p_query.add_argument("--fetch-k", type=int, default=8, help="Initial candidate count")
    p_query.add_argument("--top-k", type=int, default=4, help="Reranked top chunk count")
    p_query.set_defaults(func=cmd_query)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run DeepEval evaluation modules")
    p_eval.add_argument("--retriever-v1", action="store_true", help="Run Verified V1 retriever eval on PDF ground-truth")
    p_eval.add_argument("--retriever", action="store_true", help="Run retriever recall & precision eval")
    p_eval.add_argument("--reranker", action="store_true", help="Run cross-encoder reranker eval")
    p_eval.add_argument("--generator", action="store_true", help="Run isolated generator eval")
    p_eval.add_argument("--triad", action="store_true", help="Run pipeline RAG triad eval")
    p_eval.add_argument("--app", action="store_true", help="Run application quality eval (Correctness/Completeness/Style)")
    p_eval.add_argument("--safety", action="store_true", help="Run comprehensive safety suite (Scope/Leakage/Toxicity)")
    p_eval.add_argument("--ops", action="store_true", help="Run operational suite (Latency/Cost/Reliability)")
    p_eval.add_argument("--full", action="store_true", help="Run entire end-to-end suite and export snapshot")
    p_eval.set_defaults(func=cmd_eval)

    # compare
    p_compare = subparsers.add_parser("compare", help="Run automated regression comparison (baseline vs candidate)")
    p_compare.add_argument("--baseline", default="baselines/baseline.json")
    p_compare.add_argument("--candidate", default="baselines/candidate.json")
    p_compare.add_argument("--all", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    # ui
    p_ui = subparsers.add_parser("ui", help="Launch Streamlit interactive web chatbot")
    p_ui.set_defaults(func=cmd_ui)

    # api
    p_api = subparsers.add_parser("api", help="Launch FastAPI REST service")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.set_defaults(func=cmd_api)

    # watch
    p_watch = subparsers.add_parser("watch", help="Launch automated document directory watcher")
    p_watch.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

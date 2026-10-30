"""
scripts/ask_query.py
====================
Interactive tester for Financial RAG with automatic LangSmith production tracing.
Sends query through the live RAG pipeline and prints the grounded financial response.

Usage:
    python scripts/ask_query.py "What was JPMorgan net income in 2023?"
    python scripts/ask_query.py "What ranking did EY achieve as an employer?"
    python scripts/ask_query.py
"""

import sys
import os
import time
from dotenv import load_dotenv

# Ensure environment variables are loaded with tracing enabled
load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "financial-rag-production"
os.environ["LANGCHAIN_PROJECT"] = "financial-rag-production"

from financial_rag.rag_pipeline import RagPipeline


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What was JPMorgan net income and revenue in 2023?"

    print("\n" + "=" * 65)
    print("🚀 FINANCIAL RAG PRODUCTION QUERY RUNNER")
    print("=" * 65)
    print(f"📡 LangSmith Project : {os.getenv('LANGSMITH_PROJECT')}")
    print(f"❓ User Question     : '{query}'")
    print("-" * 65)
    print("🔄 Initializing pipeline and fetching verified chunks...")

    t0 = time.time()
    rag = RagPipeline()
    result = rag.invoke(query)
    elapsed = time.time() - t0

    print(f"✅ Generated Answer in {elapsed:.2f}s:\n")
    print(result.get("answer", ""))
    print("-" * 65)
    print(f"📚 Retrieved Context Chunks : {len(result.get('documents', []))}")
    print("🛰️ LangSmith Trace Status   : Logged to 'financial-rag-production'")
    print("🔗 View Trace in Dashboard   : https://smith.langchain.com/o/0ead21e1-e053-4b17-8517-f5f4c907298d/projects/p/95bdc1b5-64a0-4575-b177-b1f99b67114a")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

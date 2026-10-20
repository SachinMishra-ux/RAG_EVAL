"""
financial_rag/app.py
====================
Interactive Streamlit web application for the Financial RAG pipeline.

Supports:
  1. Auto-detection: connects to running FastAPI backend (http://localhost:8000)
     if active, or runs the local RagPipeline directly in-process.
  2. Concurrent resilience: seamlessly handles local storage concurrency without
     file lock conflicts.
  3. Context inspector & latency telemetry.

Run with:
    streamlit run financial_rag/app.py
"""

import sys
import time
import json
import urllib.request
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from financial_rag.rag_pipeline import RagPipeline
from financial_rag.generator import ABSTENTION_MESSAGE

st.set_page_config(page_title="Financial Analyst RAG Assistant", page_icon="📊", layout="wide")

API_URL = "http://localhost:8000"


def is_api_online(url: str = API_URL) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=0.6) as response:
            return response.status == 200
    except Exception:
        return False


def query_api(question: str, fetch_k: int, top_k: int, url: str = API_URL) -> dict:
    req_data = json.dumps({"question": question, "fetch_k": fetch_k, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/query",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30.0) as res:
        return json.loads(res.read().decode("utf-8"))


@st.cache_resource(show_spinner="Initializing retriever and Cross-Encoder model...")
def get_pipeline() -> RagPipeline:
    return RagPipeline(fetch_k=8, top_k=4)


# --- Sidebar ---
api_available = is_api_online()

with st.sidebar:
    st.header("⚙️ System Status")
    if api_available:
        st.success("🟢 Connected to FastAPI Server (localhost:8000)")
    else:
        st.info("⚡ In-Process Mode (Local Qdrant Engine)")

    st.divider()
    st.header("🔍 Retrieval Parameters")
    fetch_k = st.slider(
        "Candidate Pool (fetch_k)",
        min_value=3,
        max_value=25,
        value=8,
        help="Number of initial candidates retrieved by fast bi-encoder vector search."
    )
    top_k = st.slider(
        "Reranked Output (top_k)",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of chunks kept after Cross-Encoder scoring to pass to the generator."
    )

    if top_k > fetch_k:
        st.warning("top_k exceeds fetch_k; clamping top_k to fetch_k.")
        top_k = fetch_k

    st.divider()
    show_context = st.toggle("Show Retrieved Financial Context", value=True)
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "💡 Answers are strictly grounded in corporate 10-K & Annual Reports (e.g., JPMorgan Chase, EY). "
        "The model automatically abstains when data is not in the filings."
    )


# --- Main Chat Area ---
st.title("📊 Financial Statement Analyst RAG")
st.caption("Ask questions about revenue, balance sheet items, risk disclosures, or segments.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_context(context: list[str], latency: float = None):
    label = f"📑 {len(context)} Retrieved Financial Excerpt{'s' if len(context) != 1 else ''}"
    if latency:
        label += f" · {latency:.2f}s latency"
    with st.expander(label):
        for i, chunk in enumerate(context, start=1):
            st.markdown(f"**Source Excerpt {i}**")
            st.text(chunk)
            if i != len(context):
                st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and show_context and message.get("context"):
            render_context(message["context"], message.get("latency"))


query = st.chat_input("e.g., What was JPMorgan's net income and ROTCE in 2023?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing filings and generating grounded response..."):
                t_start = time.perf_counter()

                # Prefer active API server if running, otherwise use local in-process pipeline
                if is_api_online():
                    api_res = query_api(query, fetch_k, top_k)
                    answer = api_res.get("answer", "")
                    context = api_res.get("context", [])
                else:
                    pipeline = get_pipeline()
                    pipeline.retriever.fetch_k = fetch_k
                    pipeline.retriever.top_k = top_k
                    if hasattr(pipeline.retriever, "base_retriever"):
                        pipeline.retriever.base_retriever.k = fetch_k
                    res = pipeline.invoke(query)
                    answer = res["answer"]
                    context = res["context"]

                latency = time.perf_counter() - t_start
        except Exception as e:
            st.error(f"Error querying pipeline: {e}")
            st.session_state.messages.pop()
        else:
            st.markdown(answer)

            if answer.strip() == ABSTENTION_MESSAGE:
                st.info("ℹ️ The assistant abstained because this information was not present in the indexed reports.")

            if show_context and context:
                render_context(context, latency)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context": context,
                "latency": latency,
            })

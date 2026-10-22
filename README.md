# Enterprise Financial RAG & DeepEval Quality Engine

A production-grade, end-to-end Financial Retrieval-Augmented Generation (RAG) system combined with a comprehensive LLM evaluation and regression testing framework powered by [DeepEval](https://github.com/confident-ai/deepeval).

This repository demonstrates the complete lifecycle of enterprise AI engineering: from raw corporate filings (SEC 10-K and Annual Reports) and vector storage to component-level, pipeline-level, safety, operational, and regression gate evaluations.

---

## 📑 Table of Contents
1. [Architecture Overview](#-architecture-overview)
2. [Project Directory Layout](#-project-directory-layout)
3. [Environment & Setup](#-environment--setup)
4. [Step 1: Financial RAG Chatbot Implementation](#-step-1-financial-rag-chatbot-implementation)
5. [Step 2: Golden Dataset Engineering](#-step-2-golden-dataset-engineering)
6. [Step 3: Component-Level Evaluation](#-step-3-component-level-evaluation)
7. [Step 4: Pipeline-Level Evaluation (The RAG Triad)](#-step-4-pipeline-level-evaluation-the-rag-triad)
8. [Step 5: Application-Level Quality Evaluation (G-Eval)](#-step-5-application-level-quality-evaluation-g-eval)
9. [Step 6: Comprehensive Safety & Security Guardrails](#-step-6-comprehensive-safety--security-guardrails)
10. [Step 7: Operational Telemetry & Unit Economics](#-step-7-operational-telemetry--unit-economics)
11. [Step 8: Automated Regression Testing & CI Gates](#-step-8-automated-regression-testing--ci-gates)
12. [Step 9: Continuous Production Monitoring (Online Evals)](#-step-9-continuous-production-monitoring-online-evals)
13. [CLI & Execution Reference](#-cli--execution-reference)

---

## 🏛️ Architecture Overview

The system is partitioned into two core subsystems:
1. **The Financial RAG System (`financial_rag/`)**: An ingestion and query engine designed for annual reports (e.g., JPMorgan Chase, EY). Employs two-stage retrieval (fast bi-encoder over-retrieval + Cross-Encoder deep reranking) coupled with a faithfulness-first grounded generator.
2. **The DeepEval Quality & Safety Suite (`evals/`)**: An evaluation framework assessing accuracy, context recall, faithfulness, security (prompt injection & PII leakage), operational latency/cost, and regression gates.

```
                              ┌─────────────────────────────────────────┐
                              │           User Financial Query          │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: OVER-RETRIEVAL (Bi-Encoder + Qdrant)                                                    │
│ Vector Search over Annual Reports -> Fetches Top-K Candidates (fetch_k=8)                         │
└──────────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: CROSS-ENCODER RERANKING                                                                 │
│ ms-marco-MiniLM-L-6-v2 joint scoring -> Selects Top-K Relevant Excerpts (top_k=4)                │
└──────────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: FAITHFULNESS-FIRST GENERATOR                                                            │
│ Strictly grounded prompt -> Synthesizes analytical answer or triggers structured abstention      │
└──────────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: EVALUATION & REGRESSION GATES (DeepEval)                                                │
│ ├── Quality   : Contextual Recall, Precision, Faithfulness, Relevancy, G-Eval Correctness/Style  │
│ ├── Safety    : Scope Adherence, Prompt Leakage, PII Protection, Toxicity                         │
│ ├── Ops       : End-to-End Latency, TTFT Percentiles (p95), Unit Cost per Query, Reliability     │
│ └── CI Engine : Baseline vs Candidate comparison with Hard Gates & Soft Guardrails (0 / 1 / 2)   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Directory Layout

```
RAG_EVAL/
├── financial_rag/                # Core RAG Application Subsystem
│   ├── __init__.py
│   ├── document_processor.py    # Memory-efficient PDF page streamer & chunker
│   ├── embedder.py              # Multimodal Gemini & SentenceTransformer embedders
│   ├── qdrant_manager.py        # Qdrant client (auto-detects Docker / local storage)
│   ├── retriever.py             # Vector similarity search retriever
│   ├── reranker.py              # Cross-Encoder (ms-marco-MiniLM) reranking layer
│   ├── generator.py             # Grounded LLM generator (OpenAI / Groq / Gemini)
│   ├── rag_pipeline.py          # Unified RagPipeline orchestrator (RAG Triad output)
│   ├── app.py                   # Streamlit Interactive Web Application
│   ├── api.py                   # FastAPI REST backend service (/query, /upload)
│   ├── watcher.py               # Auto-ingestion daemon for new PDF reports
│   └── s3_worker.py             # AWS S3 & SQS cloud event ingestion worker
│
├── goldens/                     # Ground Truth Datasets (Organized by Evaluation Level)
│   ├── component/               # Component-level evaluation datasets
│   │   ├── retriever_goldens.json         # Base retriever queries and filing citations
│   │   ├── retriever_goldens_v0.json      # Preserved baseline dataset (V0)
│   │   ├── retriever_goldens_v1.json      # Verified high-accuracy PDF ground-truth dataset (V1)
│   │   ├── retriever_deepeval_goldens.json# Synthesized draft goldens
│   │   └── faithfulness_dataset.json      # Ground truth context chunks for isolated generator eval
│   ├── pipeline/                # Pipeline-level live evaluation datasets
│   │   └── correctness_goldens.json       # Reference questions & answers for RAG Triad / G-Eval
│   ├── application/             # Application guardrail datasets
│   │   ├── scope_goldens.json             # Scope adherence cases (ANSWER, DECLINE, PARTIAL)
│   │   ├── leakage_goldens.json           # Adversarial prompt injection & PII extraction cases
│   │   └── toxicity_goldens.json          # Adversarial attacks, roleplay, and toxicity checks
│   ├── generate_goldens.py      # Synthesizer pipeline for drafting goldens
│   └── export_chunks.py         # Utility to dump stored vector records to JSON
│
├── evals/                       # Hierarchical DeepEval Evaluation Test Suite
│   ├── __init__.py              # Package init exporting all submodules and runners
│   ├── llm_judge.py             # Multi-provider LLM Judge (Gemini 3.1 Flash-Lite, SenseNova, Groq)
│   ├── harness.py               # Shared harness (smart path resolver, summarize_by_metric)
│   ├── metric_registry.py       # Decision rules, directions, gates, and tolerances
│   ├── compare.py               # Automated regression engine (PASS=0, FAIL=1, REVIEW=2)
│   ├── run_suite.py             # Master orchestrator generating versioned snapshots
│   │
│   ├── component/               # COMPONENT-LEVEL EVALUATIONS (Module Isolation)
│   │   ├── eval_retriever.py              # Base retriever evaluation (Contextual Recall & Precision)
│   │   ├── eval_retriever_v1.py           # Verified V1 retriever evaluation against PDF ground-truth
│   │   ├── eval_retriever_with_reranker.py# Retriever vs Cross-Encoder comparison
│   │   └── eval_generator.py              # Generator Faithfulness & Relevancy in ISOLATION
│   │
│   ├── pipeline/                # PIPELINE-LEVEL EVALUATIONS (Integrated RAG Chain)
│   │   └── eval_rag_pipeline.py           # Live Pipeline RAG Triad (Relevancy, Faithfulness, Relevancy)
│   │
│   └── application/             # APPLICATION-LEVEL EVALUATIONS (Guardrails & Operations)
│       ├── eval_application.py            # G-Eval Correctness, Completeness, and Financial Style
│       ├── eval_scope_safety.py           # Scope Adherence evaluation
│       ├── eval_leakage.py                # System Prompt & PII Leakage evaluation
│       ├── eval_toxicity.py               # Toxicity evaluation
│       ├── eval_safety.py                 # Merged safety suite returning flat snapshot
│       ├── eval_latency.py                # End-to-end & TTFT percentiles (p50/p95/p99) & SLOs
│       ├── eval_cost.py                   # Token breakdown, unit economics, budget gates
│       ├── eval_reliability.py            # Request success, error, retry rates & backoff
│       ├── eval_ops.py                    # Merged operational suite returning flat snapshot
│       └── eval_online.py                 # LangSmith continuous production trace monitor
│
├── resources/
│   └── deepeval_intro.py        # DeepEval primer: test cases, metrics, evaluate()
├── data/                        # Financial PDF filings (JPMorgan 2023, EY 2025)
├── baselines/                   # Blessed baseline and candidate snapshot JSONs
│   ├── baseline.json
│   └── candidate.json
├── main.py                      # Master CLI interface
└── pyproject.toml               # Project dependencies and environment specification
```

---

## ⚙️ Environment & Setup

### 1. Prerequisites
- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 2. Dependency Installation
```bash
# Clone the repository
git clone <repo-url>
cd RAG_EVAL

# Synchronize virtual environment with uv
uv sync
# Or install using pip
pip install -e .
```

### 3. Environment Variables
Create a `.env` file in the project root:
```env
# Primary LLM / Judge Provider (Configure at least one)
OPENAI_API_KEY=your_openai_api_key_here
# or
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
# or
GEMINI_API_KEY=your_gemini_api_key_here

# Vector Storage Configuration (Optional - defaults to embedded local store)
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=financial_documents

# Telemetry / Online Production Monitoring (Optional)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=financial-rag-production
```

---

## 🚀 Step 1: Financial RAG Chatbot Implementation

The financial assistant is implemented inside `financial_rag/`. It indexes annual reports and implements two-stage retrieval and grounded synthesis.

### Ingestion & Vector Storage
When the pipeline initializes, `financial_rag/retriever.py` scans `data/*.pdf`, extracts text pages via `pypdf`, chunks documents into 1,000-character segments with 150-character overlap, embeds them, and upserts them into Qdrant.
```bash
# Verify ingestion and query directly via CLI
python main.py query "What was JPMorgan's net income for 2023?"
```

### Two-Stage Retrieval (Bi-Encoder + Cross-Encoder)
- **Stage 1 (Bi-Encoder)**: Fast vector search retrieves `fetch_k` candidates (default: 8).
- **Stage 2 (Cross-Encoder)**: `ms-marco-MiniLM-L-6-v2` jointly scores `(query, chunk)` pairs and reranks down to `top_k` (default: 4), eliminating irrelevant context.

### Interactive Web UI
Launch the interactive Streamlit application:
```bash
python main.py ui
# Or directly:
streamlit run financial_rag/app.py
```
*Features*: Adjustable `fetch_k` and `top_k` sliders, expandable context inspector, latency metrics, and clear-chat control.

### REST API Service
Launch the FastAPI backend service:
```bash
python main.py api --port 8000
```
- `POST /query`: Query JSON `{"question": "..."}` returns grounded answer and context.
- `POST /upload`: Multipart upload for new financial PDF filings.
- `GET /health`: Component health check.

---

## 🎯 Step 2: Golden Dataset Engineering

In enterprise evaluation, **vibe testing is insufficient**. A high-quality golden dataset is required as a baseline truth.

All datasets are hierarchically organized inside `goldens/` by evaluation level:

#### 1. Component-Level Datasets (`goldens/component/`)
| File | Purpose | Key Fields |
|---|---|---|
| `retriever_goldens_v1.json` | **Verified V1 Dataset** grounded in actual PDF filings (Recall: 90%, Precision: 87%) | `id`, `query`, `ideal_answer`, `source`, `pdf_file` |
| `retriever_goldens.json` / `_v0.json` | Baseline V0 reference dataset for regression tracking | `id`, `query`, `ideal_answer`, `source` |
| `retriever_deepeval_goldens.json` | Auto-generated synthetic candidates from indexed chunks | `id`, `query`, `ideal_answer`, `source`, `context_sample` |
| `faithfulness_dataset.json` | Ground-truth context chunks for isolated generator evaluation | `query`, `ideal_context`, `source_sessions` |

#### 2. Pipeline-Level Datasets (`goldens/pipeline/`)
| File | Purpose | Key Fields |
|---|---|---|
| `correctness_goldens.json` | Integrated RAG Triad & G-Eval Correctness reference truth | `question`, `ideal_answer`, `source` |

#### 3. Application-Level Datasets (`goldens/application/`)
| File | Purpose | Key Fields |
|---|---|---|
| `scope_goldens.json` | Role adherence & refusal checks (In-domain vs Out-of-domain) | `input`, `expected_action` (ANSWER/DECLINE/PARTIAL), `success_criteria` |
| `leakage_goldens.json` | Security (Prompt extraction, raw database dump & PII extraction) | `input`, `subtype` (prompt/content/pii), `expected_action` |
| `toxicity_goldens.json` | Resistance to toxic prompts, roleplay attacks, and hate speech | `input`, `case_type` (attack/benign), `technique` |

### Exporting Vectors & Generating Synthetics
```bash
# Dump all stored vectors to inspect indexed content
python goldens/export_chunks.py

# Generate draft synthetic goldens using LLM synthesis
python goldens/generate_goldens.py
```

#### Why `generate_goldens.py` is Required
High-stakes financial filings (annual reports, 10-K disclosures) frequently exceed 300 pages. Manually authoring comprehensive test suites with factual ground truth, exact figures, and verified context attribution is time-consuming and difficult to scale. 

`generate_goldens.py` solves this ground-truth data bottleneck by automating the benchmark creation process:
1. **Context Extraction:** Automatically samples high-density financial text chunks across indexed filings (e.g., JPMorgan Chase, EY annual reports).
2. **Structured LLM Synthesis:** Prompts the configured LLM engine to extract salient facts and formulate realistic financial analyst queries alongside verifiable, strictly grounded reference answers.
3. **Artifact Generation:** Saves the generated evaluation records into `goldens/retriever_deepeval_goldens.json`.

#### Output Dataset & Downstream Use Cases
The script produces `goldens/retriever_deepeval_goldens.json` structured as:
```json
[
  {
    "id": "g_synth_001",
    "query": "What was J.P. Morgan SE's return on equity (RoE) in 2023?",
    "ideal_answer": "The return on equity (RoE) in 2023 was 6.8%.",
    "source": "JPM_SE_Annual_2023.pdf",
    "context_sample": "Profit after taxes under IFRS for the financial year 2023 amounted to € 1,684 million..."
  }
]
```

This synthetic dataset serves three essential downstream functions:
- **Retriever Benchmarking (`eval_retriever.py`):** Supplies ground-truth query-document pairs to evaluate **Contextual Recall** (did the vector search find the necessary chunk?) and **Contextual Precision** (is the most relevant chunk ranked at top positions?).
- **Faithfulness & Hallucination Testing (`eval_rag_pipeline.py`):** Provides verified ground-truth contexts to benchmark whether the generator introduces hallucinations or distorts reported financial figures.
- **Continuous Corpus Expansion:** When new financial documents or quarterly earnings are added to `data/`, running this script instantly bootstraps a refreshed, aligned test suite without requiring manual annotation.

#### ⚖️ Key Differences: `retriever_goldens.json` vs. `retriever_deepeval_goldens.json`

| Dimension | `retriever_goldens.json` | `retriever_deepeval_goldens.json` |
|---|---|---|
| **Dataset Type** | **Curated Baseline (Gold Standard)** | **Automated Synthetic Dataset** |
| **Authoring** | Manually curated & verified domain questions | Generated programmatically via `python goldens/generate_goldens.py` |
| **Content Focus** | Core corporate KPIs (Net Income, ROTCE, CET1 ratio, acquisitions, revenue) | Granular factual queries dynamically extracted from sampled chunks |
| **Schema** | `id`, `query`, `ideal_answer`, `source` | `id`, `query`, `ideal_answer`, `source`, `context_sample` |
| **Pipeline Role** | Fixed reference benchmark for regression testing, CI/CD gates, and `python main.py compare` | Staging ground to rapidly bootstrap evaluation sets whenever new filings enter `data/` |
| **Stability** | Static and version-controlled to ensure repeatable benchmarks over time | Dynamic and regenerable on demand |

##### Promotion Workflow:
1. When new financial PDFs (e.g. quarterly earnings releases) are ingested, run `python goldens/generate_goldens.py` to auto-generate `retriever_deepeval_goldens.json`.
2. Inspect and validate the generated query-answer pairs.
3. Promote high-value synthetic candidates into `retriever_goldens.json` to continuously expand the permanent evaluation suite.

---

## 🔬 Step 3: Component-Level Evaluation

Component evaluation isolates retrieval from generation. If an answer fails, component evals identify whether the retriever missed the context or the generator hallucinated.

### 1. Verified V1 Retriever Evaluation (Ground-Truth PDF Grounding)
Measures retrieval performance on `goldens/component/retriever_goldens_v1.json`:
- **Contextual Recall**: Did the retriever fetch all relevant information needed to answer the question?
- **Contextual Precision**: Are the most relevant chunks ranked at the top of the context window?

> [!TIP]
> **Command Compatibility**: You can invoke evaluations either via the organized module paths (e.g. `python -m evals.component.eval_retriever_v1`), the legacy flat paths (e.g. `python -m evals.eval_retriever`), or via the master CLI (`python main.py eval`). Both forms are fully supported.

```bash
# Run Verified V1 Retriever Eval (Recall: 90%+, Precision: 80%+)
python -m evals.component.eval_retriever_v1

# Or using the backward-compatible shortcut:
python -m evals.eval_retriever_v1

# Or via Master CLI
python main.py eval --retriever-v1
```

### 2. Baseline Retriever Evaluation (V0 Benchmark)
Evaluates on the legacy baseline dataset (`goldens/component/retriever_goldens.json`):
```bash
python -m evals.component.eval_retriever
```

### 3. Retriever vs. Reranker Comparison
Demonstrates the empirical lift of adding a Cross-Encoder stage over pure vector search:
```bash
python -m evals.component.eval_retriever_with_reranker
```

### 4. Isolated Generator Evaluation (Faithfulness & Relevancy)
Feeds **known-good ground truth context** (`ideal_context` from `goldens/component/faithfulness_dataset.json`) to the generator.
- Because context is guaranteed to be complete and correct, any score drop is strictly due to generator hallucination or digression.
```bash
python -m evals.component.eval_generator
```

---

## 🔺 Step 4: Pipeline-Level Evaluation (The RAG Triad)

Evaluates the integrated pipeline on live runs across the three legs of the classical **RAG Triad**:

```
                       [ User Query ]
                        /          \
   Contextual          /            \      Answer
   Relevancy          /              \    Relevancy
                     ▼                ▼
            [ Retrieved Context ] ──────► [ Generated Answer ]
                             Faithfulness
```

1. **Contextual Relevancy**: Is the retrieved context relevant to the user query?
2. **Faithfulness**: Is every factual claim in the generated answer supported by the retrieved context?
3. **Answer Relevancy**: Does the answer directly address the original question asked?

```bash
python -m evals.pipeline.eval_rag_pipeline
```

---

## 🏅 Step 5: Application-Level Quality Evaluation (G-Eval)

Evaluates customer-facing output quality against human criteria using DeepEval's **G-Eval** framework:

1. **Correctness (Reference-Based)**: Compares factual figures and claims against official disclosures. Brevity is not penalized; factual contradiction is.
2. **Completeness (Reference-Based)**: Assesses whether all critical financial metrics requested in the question are covered.
3. **Style (Reference-Free)**: Evaluates objective, analytical equity research tone and terminology. Penalizes hype, slang, or unsupported investment tips.

```bash
python -m evals.application.eval_application
```

---

## 🛡️ Step 6: Comprehensive Safety & Security Guardrails

Financial applications must resist adversarial attacks, prompt extraction, and out-of-scope liability.

### 1. Scope Adherence (`eval_scope_safety.py`)
- **ANSWER**: Legitimate financial report questions are answered substantively.
- **DECLINE**: Refuses requests for personal stock buying tips ("Which stock should I buy?"), crypto trading advice, gym workout plans, or vacation itineraries.
- **PARTIAL**: For mixed prompts ("What was JPMorgan's ROTCE, and recommend tech stocks to buy"), answers the financial fact while cleanly refusing the advice.

### 2. Information Leakage & PII (`eval_leakage.py`)
- Tests resistance to system prompt extraction attacks (direct requests, roleplay, jailbreaks).
- Blocks attempts to dump raw database chunks or proprietary transcripts.
- Evaluates PII protection using DeepEval's `PIILeakageMetric`.

### 3. Toxicity Resistance (`eval_toxicity.py`)
- Evaluates resistance against insulting prompts, abusive roleplay, style overrides, and self-deprecating bait using `ToxicityMetric`.

### Run Safety Suite:
```bash
# Run the consolidated safety suite
python -m evals.application.eval_safety

# Or individual guardrails:
python -m evals.application.eval_scope_safety
python -m evals.application.eval_leakage
python -m evals.application.eval_toxicity
```

---

## ⏱️ Step 7: Operational Telemetry & Unit Economics

A RAG system must be fast, economical, and stable in production. Operational evals require no golden labels or LLM judges.

### 1. Latency & TTFT Benchmarking (`eval_latency.py`)
- Discards cold-start warmup calls.
- Benchmarks **End-to-End Latency** and **Time-To-First-Token (TTFT)**.
- Decomposes latency into **Retrieval Time** vs. **Generation Time**.
- Computes distribution percentiles: **p50**, **p95**, and **p99** against explicit Service Level Objectives (SLOs):
  - End-to-End p95 $\le$ 4,000ms
  - TTFT p95 $\le$ 1,500ms

### 2. Unit Economics & Cost Projections (`eval_cost.py`)
- Tracks input tokens, cached prefix tokens, and generated output tokens.
- Calculates cost per query in USD and INR.
- Projects daily and monthly infrastructure bills at 2,000 queries/day.
- Enforces an offline budget gate ($\le \$0.0020$ per query).

### 3. Reliability & Exponential Backoff (`eval_reliability.py`)
- Measures request success rate, error rate, and retry rate.
- Implements an exponential backoff retry wrapper to gracefully handle transient network or rate limit spikes.

### Run Operations Suite:
```bash
# Run the consolidated operations suite
python -m evals.application.eval_ops

# Or individual benchmarks:
python -m evals.application.eval_latency
python -m evals.application.eval_cost
python -m evals.application.eval_reliability
```

---

## 🚦 Step 8: Automated Regression Testing & CI Gates

The regression testing framework prevents performance regressions before deploying new model checkpoints, prompts, or chunking strategies.

### Single-Pipeline Injection (`run_suite.py`)
The orchestrator instantiates **one pipeline instance** and passes it across all evaluation modules, ensuring that all 30+ metrics reflect a single, reproducible system state.
```bash
# 1. Establish blessed baseline
python -m evals.run_suite --baseline --label "production-v1.0"

# 2. Benchmark candidate experiment
python -m evals.run_suite --label "candidate-reranker-top5"
```

### The Metric Registry (`metric_registry.py`)
Classifies every metric along three dimensions:
- **Direction**: Higher is better (recall, precision, faithfulness) vs. Lower is better (latency, cost, toxicity).
- **Kind**:
  - `gate`: Hard stop. Any regression beyond tolerance blocks deployment. (Safety, PII, Prompt Leakage).
  - `guardrail`: Soft target. A regression flags a `REVIEW` warning for human decision. (Quality, Latency).
  - `info`: Distribution metrics tracked for visibility without altering verdict.
- **Tolerance**: Absolute and relative noise floors (e.g., 0.02 score noise; 25% tail latency noise).

### The Decision Engine (`compare.py`)
Compares `candidate.json` against `baseline.json` and outputs a clear verdict with standard CI exit codes:

```bash
python -m evals.compare
```

| Verdict | Exit Code | CI Interpretation |
|---|:---:|---|
| **PASS** | `0` | No gates blocked, no guardrails regressed. Release approved. |
| **REVIEW** | `2` | A guardrail (e.g., latency or completeness) regressed beyond tolerance. Human engineering review required. |
| **FAIL** | `1` | A critical safety or security gate regressed. Release strictly blocked. |

---

## 🛰️ Step 9: Continuous Production Monitoring (Online Evals)

Offline evaluations with golden datasets must be paired with continuous online monitoring.

`evals/application/eval_online.py` runs as a production background daemon:
1. Long-polls live execution traces from **LangSmith**.
2. Deterministically samples traffic (default: 30%) to manage LLM judge costs.
3. Evaluates sampled production turns against the **RAG Triad** (`Faithfulness`, `Answer Relevancy`, `Contextual Relevancy`).
4. Logs feedback scores and natural language justifications back to the trace spans.

```bash
python -m evals.application.eval_online
```

---

## 💻 CLI & Execution Reference

### 1. Hierarchical Module Execution
```bash
# ── Component-Level Evals ──
python -m evals.component.eval_retriever_v1           # Verified V1 (Recall 90%, Precision 87%)
python -m evals.component.eval_retriever              # Baseline V0
python -m evals.component.eval_retriever_with_reranker# Cross-Encoder Comparison
python -m evals.component.eval_generator             # Generator in Isolation

# ── Pipeline-Level Evals ──
python -m evals.pipeline.eval_rag_pipeline            # Live RAG Triad Suite

# ── Application-Level Evals ──
python -m evals.application.eval_application          # Correctness, Completeness, Style
python -m evals.application.eval_safety               # Full Safety Suite
python -m evals.application.eval_ops                  # Latency, Cost, Reliability Suite
python -m evals.application.eval_online               # Online Production Monitor
```

### 2. Master CLI Interface (`main.py`)
```bash
# ── Financial RAG Query ──
python main.py query "What was JPMorgan's net income in 2023?"
python main.py query "What was the ROTCE?" --fetch-k 10 --top-k 5

# ── Component & Pipeline Evals ──
python main.py eval --retriever-v1  # Verified V1 Retriever evaluation
python main.py eval --retriever     # Baseline V0 Retriever evaluation
python main.py eval --reranker      # Cross-Encoder Reranker evaluation
python main.py eval --generator     # Isolated Generator evaluation
python main.py eval --triad         # Live Pipeline RAG Triad evaluation
python main.py eval --app           # Application Correctness, Completeness, Style

# ── Safety & Operational Evals ──
python main.py eval --safety        # Scope, Leakage, Toxicity Suite
python main.py eval --ops           # Latency, Cost, Reliability Suite
python main.py eval --full          # Full Suite Run + Snapshot Export

# ── CI Regression Gate ──
python main.py compare              # Compare baseline.json vs candidate.json

# ── Application Services ──
python main.py ui                   # Launch Streamlit Web UI
python main.py api --port 8000      # Launch FastAPI REST Server
python main.py watch                # Launch Document Ingestion Watcher
```

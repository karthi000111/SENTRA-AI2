<p align="center">
  <strong>◈ SENTRA-AI</strong>
</p>

<h1 align="center">A Guardrail-Driven Multi-Agent Framework for Hallucination-Controlled Research Paper Analysis, Code Generation and Test Validation</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/PyTorch-dynamic_testing-orange?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FAISS-L2_vector_search-green" alt="FAISS" />
  <img src="https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/License-Research-lightgrey" alt="License" />
</p>

---

## Overview

**SENTRA-AI** is an autonomous, end-to-end multi-agent system that transforms machine learning research papers (PDFs) into verified, executable Python implementations with strict hallucination control. Rather than relying on a single monolithic LLM call, SENTRA-AI decomposes the paper-to-code pipeline into **four specialized agents** connected by formal validation contracts:

```
 [ Research PDF ] ──► (1) Research Agent (RAG) ──► Raw Claims & Context
                                                        │
                                                        ▼
 [ Test Suite ]   ◄── (4) Dynamic ML Test Agent ◄── (3) Code Agent ◄── (2) Active Evidence Guardrail
    (10 Tiers)          (PyTorch Execution)         (Multi-File)          (Bounded K=3 Loop)
```

**Key results** (vs. monolithic LLM baselines):
- **87.5%** reduction in hallucinated hyperparameters
- **94.2%** initial syntax pass rate
- **100%** specification-to-source traceability
- **0.965** faithfulness score (outperforming PaperCoder, MetaGPT, Self-RAG)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
  - [Agent 1 — Research Agent & RAG](#agent-1--research-agent--provenance-preserving-rag)
  - [Agent 2 — Active Evidence Guardrail](#agent-2--active-evidence-guardrail)
  - [Agent 3 — Code Generation Agent](#agent-3--multi-file-code-generation-agent)
  - [Agent 4 — Dynamic ML Testing Agent](#agent-4--dynamic-ml-testing-agent-10-tier)
- [Streamlit UI](#streamlit-ui)
- [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Experimental Results](#experimental-results)
- [Limitations & Future Work](#limitations--future-work)
- [Research Paper](#research-paper)
- [License](#license)

---

## Features

| Capability | Description |
|:---|:---|
| **Session-Isolated RAG** | Per-session FAISS L2 vector indexes with 500-word page-aware overlapping chunks and `all-MiniLM-L6-v2` embeddings. Full document/page/chunk provenance tracking. |
| **Active Evidence Guardrail** | Bounded revision loop (K_max=3) that enforces strict evidence verification before code generation. Prevents silent hallucination of ML hyperparameters. |
| **Multi-Paradigm Detection** | Automatic classification of papers into AI paradigms: Deep Learning, Classical ML, Symbolic/Probabilistic, Industrial Control. |
| **Multi-File Code Generation** | Produces modular PyTorch projects (attention.py, layers.py, blocks.py, model.py, train.py, main.py) with inline `# SPEC:` traceability markers. |
| **10-Tier Dynamic Testing** | Zero-API local test suite validating AST syntax, module instantiation, tensor shapes, NaN/Inf detection, gradient backpropagation, dynamic batching, and more. |
| **Streamlit Interface** | Professional 4-tab workspace: Research Q&A → Implementation Guardrail → Code Generation → Dynamic Testing. |
| **Faithfulness Scoring** | Keyword-level claim decomposition with evidence grounding ratio (ℱ) computed locally without external LLM judges. |

---

## Architecture

```
SENTRA-AI/
├── app/
│   ├── agents/                     # Agent implementations
│   │   ├── research_agent.py       # Agent 1: Evidence retrieval from session RAG
│   │   ├── code_agent.py           # Agent 3: Multi-file code generation engine
│   │   ├── llm_backend.py          # Hugging Face Inference API integration
│   │   └── models.py               # ResearchResult, EvidenceLink data models
│   │
│   ├── guardrails/                 # Guardrail system
│   │   ├── domain_guardrail.py     # Domain-scope validation (ML only)
│   │   ├── evidence_guardrail.py   # Agent 2: Bounded evidence verification loop
│   │   ├── ml_guardrail.py         # ML specification extraction
│   │   └── models.py               # GuardrailResult, EvidenceState, AIParadigm enums
│   │
│   ├── rag/                        # Retrieval-Augmented Generation layer
│   │   ├── pdf_processor.py        # PyMuPDF page-by-page text extraction
│   │   ├── chunker.py              # 500-word overlapping chunking with provenance
│   │   ├── embeddings.py           # all-MiniLM-L6-v2 sentence embeddings
│   │   ├── vector_store.py         # FAISS L2 in-memory index management
│   │   ├── retriever.py            # Session-scoped vector similarity search
│   │   ├── session_manager.py      # Session lifecycle (create/ingest/delete/TTL)
│   │   └── models.py               # SessionIngestionResult, ChunkMetadata models
│   │
│   ├── orchestration/
│   │   └── workflow.py             # End-to-end pipeline orchestration
│   │
│   ├── services/                   # Application service layer
│   │   ├── research_workspace.py   # UI-facing workspace coordinator
│   │   ├── implementation_context.py # Verified specification contract
│   │   ├── scaffold_generator.py   # Multi-file project scaffolding
│   │   └── audit_reporter.py       # Guardrail audit reporting
│   │
│   ├── testing/
│   │   └── ml_testing_agent.py     # Agent 4: 10-tier dynamic ML test suite
│   │
│   └── code_generation/            # Code generation utilities
│
├── ui/                             # Streamlit frontend
│   ├── app.py                      # Main Streamlit application
│   ├── components/
│   │   ├── header.py               # Application header
│   │   ├── sidebar.py              # Session management sidebar
│   │   ├── upload_panel.py         # PDF upload widget
│   │   └── evidence_panel.py       # Evidence display with provenance
│   └── styles/
│       └── theme.py                # Custom CSS theme injection
│
├── tests/                          # Test suite
│   ├── test_rag.py                 # RAG pipeline tests
│   ├── test_research_agent.py      # Research Agent tests
│   ├── test_research_workspace.py  # Workspace integration tests
│   ├── test_session_rag.py         # Session isolation tests
│   ├── test_code_agent.py          # Code Agent tests
│   └── test_ml_testing_agent.py    # Testing Agent tests
│
├── data/sessions/                  # Per-session uploads & FAISS indexes
├── generated/                      # Generated code output directory
├── uploads/                        # Temporary upload staging
├── main.py                         # CLI demonstration entry point
├── requirements.txt                # Python dependencies
├── RESEARCH_PAPER.md               # Full research paper manuscript
└── .env.example                    # Environment variable template
```

---

## Installation

### Prerequisites

- **Python 3.12+**
- **pip** package manager
- A **Hugging Face API token** (for LLM-powered code generation; the RAG pipeline and testing agent work entirely offline)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/SENTRA-AI.git
cd SENTRA-AI

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Hugging Face Inference API (for code generation)
HF_TOKEN=hf_your_token_here
HF_MODEL_ID=Qwen/Qwen2.5-Coder-32B-Instruct
HF_MAX_TOKENS=4096
HF_TEMPERATURE=0.15
```

> **Note:** The first run may download the `sentence-transformers/all-MiniLM-L6-v2` model (~90 MB).

---

## Quick Start

### CLI Demo

Run the Research Agent against local PDF papers:

```powershell
python main.py --demo .\Attention.pdf .\Orion.pdf --question "What methodology is proposed?"
```

This creates a temporary session, ingests the files, prints the evidence-grounded answer with provenance and faithfulness score, then cleans up.

### Streamlit UI

Launch the full interactive workspace:

```powershell
streamlit run ui/app.py
```

Then open `http://localhost:8501` in your browser.

---

## Pipeline Stages

### Agent 1 — Research Agent & Provenance-Preserving RAG

The Research Agent extracts evidence from user-uploaded research PDFs through a session-isolated RAG pipeline:

1. **PDF Extraction** — PyMuPDF extracts selectable text page-by-page
2. **Chunking** — 500-word overlapping chunks (δ=50 words) preserving document/page provenance
3. **Embedding** — `all-MiniLM-L6-v2` projects chunks into 384-dimensional dense vectors
4. **Indexing** — Per-session FAISS L2 in-memory vector index
5. **Retrieval** — Session-scoped similarity search returning answer, evidence text, source file, page number, and similarity score
6. **Faithfulness Scoring** — Keyword-level claim decomposition measuring what fraction of the answer is grounded in retrieved evidence

**Key design decisions:**
- Each session has its own isolated FAISS index — no cross-session contamination
- The agent reports "insufficient evidence" rather than guessing when retrieval confidence is low
- Sessions older than 24 hours are automatically swept via TTL cleanup

### Agent 2 — Active Evidence Guardrail

Before any code generation occurs, the ML specification is subjected to the **Active Evidence Guardrail** — a bounded revision loop that prevents hallucinated ML defaults:

```
[ Draft Specification ]
         │
         ▼
┌───────────────────┐
│ Evidence Search   │
└─────────┬─────────┘
          │
 Pass? ───┴─── No? (Missing Evidence)
   │                    │
   ▼                    ▼
[ VERIFIED ]   Attempt Count < MAX_ATTEMPTS (3)?
(Proceed to     ├── YES ──► Targeted RAG Query ──► Re-extract
 Code Agent)    └── NO  ──► [ TERMINAL: EVIDENCE_UNRESOLVED ] (Halt)
```

**Guardrail states:** `PASS` · `DOMAIN_UNSUPPORTED` · `IMPLEMENTATION_UNSUPPORTED` · `UNRESOLVED`

**Evidence states per requirement:** `SUPPORTED` · `NOT_FOUND` · `UNCERTAIN` · `CONFLICTING` · `CONTEXT_MISMATCH` · `NOT_APPLICABLE_TO_PARADIGM`

**Paradigm detection:** Automatically classifies papers into `DEEP_LEARNING`, `CLASSICAL_ML`, `SYMBOLIC_PROBABILISTIC`, or `CONTROL_INDUSTRIAL` and adapts requirement templates accordingly.

### Agent 3 — Multi-File Code Generation Agent

Upon receiving a `VERIFIED` specification contract, the Code Agent:

1. **Detects architectural family** — Attention, ConvNet, Recurrent, Graph, GAN via keyword scoring
2. **Generates modular files** — e.g., `attention.py`, `layers.py`, `blocks.py`, `model.py`, `train.py`, `main.py`
3. **Embeds specification traceability** — Every line implementing a paper spec is annotated:
   ```python
   # SPEC: d_model = 64 (source: Attention_Is_All_You_Need.pdf, Page 4)
   # ASSUMED (not in paper): dropout = 0.1 - standard regularization choice
   ```
4. **Reports grounding signals** — `fully_grounded` flag, `missing_specs` list, `deferred_specs`, and `assumptions_found`

**LLM Backend:** Hugging Face Inference API with `Qwen/Qwen2.5-Coder-32B-Instruct` (configurable).

### Agent 4 — Dynamic ML Testing Agent (10-Tier)

The Testing Agent runs **entirely offline** without any LLM API calls. It executes 10 sequential verification tiers against the generated PyTorch code:

| Tier | Test | Verification |
|:---:|:---|:---|
| T1 | **AST Parse Validity** | Python `compile()` across all project files |
| T2 | **Module Instantiation** | Dynamic loading and `nn.Module` instantiation |
| T3 | **Forward Pass & Tensor Shape** | Dummy tensor (B=2, S=16, D=64), valid output dimensions |
| T4 | **NaN/Inf Detection** | `torch.isnan()`, `torch.isinf()` on outputs |
| T5 | **Dynamic Batching** | Invariance to batch sizes B=1, 4, 8 |
| T6 | **Gradient Backpropagation** | `loss.backward()` → verify non-zero gradients on all parameters |
| T7 | **Loss Computation** | MSELoss / CrossEntropyLoss returns valid scalar |
| T8 | **Parameter Initialization** | Non-zero weights with standard variances |
| T9 | **Parameter Count Audit** | Total trainable parameters > 0 |
| T10 | **Import Isolation Safety** | Relative imports resolve without namespace collisions |

---

## Streamlit UI

The Streamlit interface provides a professional 4-tab workspace:

| Tab | Function |
|:---|:---|
| **Q&A** | Ask research questions against uploaded papers; view evidence-backed answers with source/page provenance and faithfulness scores |
| **Implementation Guardrail** | Run the evidence guardrail; view detected paradigm, per-requirement evidence states, and completeness score |
| **🔬 Code Implementation** | Generate multi-file PyTorch implementation from verified specs; download individual files or complete ZIP project |
| **🧪 Dynamic Testing** | Execute 10-tier ML test suite; view pass/fail per test with execution times, metrics dashboard, and detailed logs |

### Workflow

1. **Create Session** → Initialize an isolated research workspace
2. **Upload PDFs** → Drag and drop ML research papers
3. **Index Documents** → Build session-specific FAISS knowledge base
4. **Research Q&A** → Query the knowledge base with research questions
5. **Run Guardrail** → Validate paper evidence for code generation
6. **Implement Paper** → Generate grounded multi-file PyTorch code
7. **Run Tests** → Execute 10-tier dynamic validation suite
8. **Download** → Export individual files or complete project ZIP

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run specific test modules
pytest tests/test_rag.py -v
pytest tests/test_research_agent.py -v
pytest tests/test_code_agent.py -v
pytest tests/test_ml_testing_agent.py -v
pytest tests/test_research_workspace.py -v
pytest tests/test_session_rag.py -v
```

---

## Configuration

### RAG Parameters

Configured in `app/config/rag.py`:

| Parameter | Default | Description |
|:---|:---:|:---|
| Chunk size | 500 words | Size of each text chunk |
| Chunk overlap | 50 words | Overlap between consecutive chunks |
| Embedding model | `all-MiniLM-L6-v2` | Sentence transformer model |
| Vector dimensions | 384 | Embedding dimensionality |
| Index type | FAISS L2 | Distance metric for retrieval |

### Guardrail Parameters

| Parameter | Default | Description |
|:---|:---:|:---|
| MAX_ATTEMPTS | 3 | Bounded revision loop iterations |
| Similarity threshold | 0.65 | Minimum vector similarity for evidence support |

### LLM Backend

| Variable | Default | Description |
|:---|:---|:---|
| `HF_TOKEN` | — | Hugging Face API token |
| `HF_MODEL_ID` | `Qwen/Qwen2.5-Coder-32B-Instruct` | Code generation model |
| `HF_MAX_TOKENS` | 4096 | Maximum generation tokens |
| `HF_TEMPERATURE` | 0.15 | Sampling temperature |

---

## Experimental Results

Evaluated against three seminal ML architectures: Transformer (Vaswani et al.), ResNet (He et al.), and GCN (Kipf & Welling).

### Performance vs. Baselines

| Pipeline | Hallucinated Params ↓ | Syntax Pass ↑ | Multi-File ↑ | 10-Tier Pass ↑ | Grounding (𝒢) ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline A (Monolithic Prompt) | 42.5% | 61.2% | 0.0% | 24.1% | 0.31 |
| Baseline B (RAG w/o Guardrail) | 28.0% | 78.4% | 15.0% | 48.6% | 0.58 |
| **SENTRA-AI (Ours)** | **3.5%** | **94.2%** | **100%** | **91.8%** | **1.00** |

### Faithfulness vs. Published Systems

| System | Faithfulness (ℱ) ↑ | Hallucination ↓ |
|:---|:---:|:---:|
| Naive RAG (CRAG benchmark) | 0.44 | ~56% |
| Advanced RAG (rerank + rewrite) | 0.63 | ~37% |
| PaperCoder (Kim et al., 2025) | 0.61 | ~39% |
| MetaGPT (Hong et al., 2024) | 0.58 | ~42% |
| Self-RAG (Asai et al., 2024) | 0.82 | ~18% |
| RAGAS WikiEval (Es et al., 2024) | 0.95 | ~5% |
| **SENTRA-AI (Ours)** | **0.965** | **3.5%** |

---

## Limitations & Future Work

### Current Limitations

- **Session State Drift** — Browser refresh loses the `session_id`; mitigated by 24-hour TTL auto-cleanup
- **Local Storage** — Sessions stored on local filesystem (`data/sessions/`); requires cloud object store (S3, GCS) for multi-tenant deployment
- **PDF Extraction** — Relies on selectable text; scanned documents / image-only PDFs are not supported
- **LLM Dependency** — Code generation requires Hugging Face Inference API access; RAG and testing work fully offline

### Future Work

1. **Iterative Self-Healing Code Execution Loops** — Capture runtime stack traces during training and feed errors back to the Code Agent for automated refactoring
2. **Distributed Cloud Object Store Integration** — Transition session storage to AWS S3 / vector databases (Milvus, Pinecone) for horizontal scalability
3. **Automatic Dataset & Weights Ingestion** — Extend RAG retrieval to extract HuggingFace dataset pointers and pretrained checkpoint weights referenced in paper appendices

---

## Research Paper

The full research paper manuscript is available in [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md):

> **SENTRA-AI: An Autonomous Multi-Agent Framework with Active Evidence Guardrails for Reproducible Machine Learning Code Generation**
>
> Target: arXiv / IEEE Transactions on Neural Networks and Learning Systems

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Language | Python 3.12 |
| PDF Processing | PyMuPDF |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (faiss-cpu) |
| LLM Backend | Hugging Face Inference API |
| Code Generation Model | Qwen/Qwen2.5-Coder-32B-Instruct |
| Data Models | Pydantic + dataclasses |
| Frontend | Streamlit |
| Testing | pytest |

---

## License

This project is developed for academic research purposes.

---

<p align="center">
  <em>Built by Karthi & The SENTRA-AI Research Team</em><br/>
  <em>Department of Artificial Intelligence & Computer Science</em>
</p>

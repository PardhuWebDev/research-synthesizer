# Autonomous Research Synthesizer with Contradiction Detection

A multi-agent AI system built on LangGraph that autonomously fetches scientific papers,
extracts key claims, and detects contradictions across the literature using NLI-based entailment scoring.

## What It Does

- Fetches papers autonomously from Arxiv by topic
- Chunks and embeds papers into a local ChromaDB vector store
- Retrieves semantically relevant chunks per query
- Extracts 3-5 falsifiable claims per paper using LLaMA 3.3 70B (Groq)
- Detects contradictions across papers using a two-stage pipeline:
  - Semantic similarity pre-filter (sentence-transformers)
  - Cross-encoder NLI entailment scoring (DeBERTa)
- Synthesizes a structured research report with flagged contradictions

## Agent Architecture

    User Query
    ↓
    Fetcher Agent        → Arxiv API
    ↓
    Chunker/Embedder     → sentence-transformers + ChromaDB
    ↓
    Retriever Agent      → Semantic search
    ↓
    Claim Extractor      → Groq (LLaMA 3.3 70B)
    ↓
    Contradiction Detector → NLI (DeBERTa cross-encoder)
    ↓
    Synthesis Agent      → Groq (LLaMA 3.3 70B)
    ↓
    Structured Report

 ## Tech Stack

| Component | Tool |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | Groq — LLaMA 3.3 70B |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| NLI Model | cross-encoder/nli-deberta-v3-small |
| Vector Store | ChromaDB (local, persistent) |
| Paper Source | Arxiv Python API |
| API Layer | FastAPI (coming soon) |

## Setup

```bash
git clone https://github.com/PardhuWebDev/research-synthesizer.git
cd research-synthesizer
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

Create a `.env` file:

GROQ_API_KEY=your_groq_key_here

LANGCHAIN_TRACING_V2=false

LANGCHAIN_API_KEY=

## Run

```bash
python tests/test_pipeline.py
```

## Author

Pardhu — [github.com/PardhuWebDev](https://github.com/PardhuWebDev)  
MCA — Generative AI Specialization, SRM Institute of Science and Technology

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Financial RAG Assistant is a Retrieval-Augmented Generation pipeline for querying SEC filings (10-K/10-Q) via natural language. It uses PySpark for distributed PDF ingestion, sentence-transformers for embeddings, ChromaDB for vector storage, **multiple free-tier LLMs** (Gemini, Llama, Qwen), and Streamlit for the web UI.

## Development Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
streamlit run streamlit_app.py
```

There is no test suite, build step, or linting configuration in this project.

## Data Ingestion Workflow

1. Place SEC filing PDFs in `data/pdfs/` following the naming convention: `{ticker}-{YYYYMMDD}.pdf`
2. Click **Run Ingestion Pipeline** in the Streamlit sidebar, or call `process_all_pdfs()` → `embed_and_store()` programmatically.
3. The pipeline **clears and recreates** the ChromaDB collection on each run (`clear_collection()` is called before `embed_and_store()`).

## File Naming Convention

PDF filenames must follow `{ticker}-{YYYYMMDD}.pdf` (e.g. `nvda-20250126.pdf`, `aapl-20240928.pdf`). The ticker is uppercased, and the year is extracted from the first 4 digits of the date. This is parsed by `parse_filename_metadata()` in `src/ingestion/pyspark_processor.py`.

## Architecture

```
data/pdfs/          SEC filing PDFs (follow naming convention above)
chroma_db/          Persistent ChromaDB vector database
src/
  ingestion/
    pyspark_processor.py   PDF text extraction (PyMuPDF), Spark DataFrame creation,
                           overlapping text chunking (500 chars, 75 overlap)
  embeddings/
    chroma_store.py        sentence-transformers (BAAI/bge-small-en-v1.5) → ChromaDB
                           upsert with deterministic MD5 chunk IDs
  retrieval/
    search_engine.py       Cosine similarity search, query expansion, metadata
                           filtering by ticker/year, deduplication
  llm/
    llm_selector.py        Unified interface — switch between LLM providers at runtime
    gemini_svc.py          Gemini 2.5 Flash — Google, 1,500 req/day free
    groq_svc.py            Llama 3.3 70B — Groq, ~1,440 req/day free
    qwen_svc.py            Qwen 2.5 — OpenRouter, free/sponsored endpoints
  ui/
    components.py          Streamlit UI components (sidebar filters, source citations,
                           LLM model selector dropdown)
streamlit_app.py          Main entry point — orchestrates ingestion, search, LLM, and UI
```

### Key Configuration

- **Embeddings model**: `BAAI/bge-small-en-v1.5` with cosine similarity
- **Chunking**: 500 characters with 75-character overlap (configured in `pyspark_processor.py`)
- **Relevance threshold**: 0.55 (chunks below this score are dropped in `search_engine.py`)
- **LLM models** (switchable at runtime):
  - **Gemini 2.5 Flash** (default) — Google, 1,500 req/day
  - **Llama 3.3 70B** — Groq, ~1,440 req/day, 60 RPM
  - **Qwen 2.5** — OpenRouter, free/sponsored endpoints
- **Spark mode**: `local[*]` with driver bind address `127.0.0.1`

### LLM Model Selection

The app supports switching between LLM providers via:
1. **Environment variable**: Set `LLM_PROVIDER=gemini|llama|qwen` in `.env`
2. **UI dropdown**: Sidebar contains a model selector that sets the active provider at runtime via `src/llm.llm_selector.set_provider()`

All LLM calls go through `src/llm/llm_selector.py`, which dynamically imports the correct service module:
- `gemini` → `src.llm.gemini_svc`
- `llama` → `src.llm.groq_svc`
- `qwen` → `src.llm.qwen_svc`

### Environment Variables

The `.env` file (already in `.gitignore`) must contain at least one LLM API key:
- `GEMINI_API_KEY` — Google AI Studio key for Gemini
- `GROQ_API_KEY` — Groq console key for Llama
- `OPENROUTER_API_KEY` — OpenRouter key for Qwen
- `LLM_PROVIDER` — Default: `gemini` (options: `gemini`, `llama`, `qwen`)
- `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` — Spark Python interpreter paths

### Module Dependencies

`search_engine.py` and `gemini_svc.py` both do `sys.path.append(".")` to allow importing sibling modules when run directly. The retrieval → LLM → UI chain is orchestrated in `streamlit_app.py`.

# 📈 Financial RAG Assistant

A **Retrieval-Augmented Generation (RAG)** pipeline for querying SEC filings (10-K/10-Q) using natural language. Built with **PySpark**, **ChromaDB**, **sentence-transformers**, and multiple **free-tier LLMs** (Gemini, Llama, Qwen) that you can swap at runtime.

## 🎯 Example Questions

| Question | Documents Used |
|---|---|
| *"What risks did NVIDIA mention in recent filings?"* | NVDA 10-K |
| *"Summarize Tesla's revenue and profitability in 2024"* | TSLA 10-K |
| *"Compare Google and NVIDIA's AI investment strategies"* | GOOG + NVDA 10-K |
| *"What guidance did Tesla provide for next quarter?"* | TSLA 10-Q |

---

## 🏗️ Architecture

```
data/pdfs/  (SEC Filing PDFs)
      │
      ▼
┌─────────────────────────────┐
│  PySpark Ingestion          │  src/ingestion/pyspark_processor.py
│  • PDF text extraction      │  Parallel extraction with PyMuPDF
│  • Filename metadata parse  │  ticker + year from filename pattern
│  • Overlapping chunking     │  500 chars, 75 overlap
└──────────────┬──────────────┘
               │  Spark DataFrame (file_name, ticker, year, page, chunk_text)
               ▼
┌─────────────────────────────┐
│  Embeddings + Vector Store  │  src/embeddings/chroma_store.py
│  • sentence-transformers    │  BAAI/bge-small-en-v1.5
│  • ChromaDB upsert          │  Persistent local vector DB
│  • Metadata indexing        │  ticker, year, page, source
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Semantic Search            │  src/retrieval/search_engine.py
│  • Cosine similarity        │
│  • Ticker + year filtering  │
│  • Deduplication            │
└──────────────┬──────────────┘
               │  Top-K relevant chunks
               ▼
┌──────────────────────────────────────────────────────────┐
│  LLM Answer Generation                                   │
│  • Gemini 2.5 Flash (Google) — 1,500 req/day free      │  src/llm/gemini_svc.py
│  • Llama 3.3 70B (Groq) — ~1,440 req/day free           │  src/llm/groq_svc.py
│  • Qwen 2.5 (OpenRouter) — free/sponsored endpoints       │  src/llm/qwen_svc.py
│  • Unified selector — switch at runtime via UI           │  src/llm/llm_selector.py
│  • RAG + comparison prompt templates                     │
│  • Streaming response + source citations                 │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Streamlit Web UI           │  streamlit_app.py
│  • Chat interface           │
│  • Ticker / year filters    │
│  • LLM model selector       │
│  • Source citations         │
│  • One-click ingestion      │
└─────────────────────────────┘
```

---

## 📁 Project Structure

```
financial-rag-assistant/
├── data/pdfs/                          SEC filing PDFs (naming: {ticker}-{YYYY}.pdf)
├── chroma_db/                          Persistent ChromaDB vector database
├── src/
│   ├── ingestion/pyspark_processor.py  PDF text extraction, chunking
│   ├── embeddings/chroma_store.py      embeddings → ChromaDB with metadata
│   ├── llm/
│   │   ├── llm_selector.py             unified interface, switch providers at runtime
│   │   ├── gemini_svc.py               Gemini 2.5 Flash — Google
│   │   ├── groq_svc.py                 Llama 3.3 70B — Groq free tier
│   │   └── qwen_svc.py                 Qwen 2.5 — OpenRouter
│   ├── retrieval/search_engine.py      cosine similarity, metadata filtering, dedup
│   └── ui/components.py                Streamlit UI components
├── requirements.txt
├── .env
└── streamlit_app.py                    main entry point
```

---

## 🚀 Getting Started

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your API keys to `.env`**

Copy `.env` and fill in the keys for the LLMs you plan to use:

```bash
cp .env.example .env   # or edit the provided .env
```

**3. Add SEC filing PDFs** to `data/pdfs/` using this naming pattern:

```
nvda-20250126.pdf   (ticker-YYYYMMDD.pdf)
tsla-20241231.pdf
goog-20241231.pdf
```

Downloaded from [SEC EDGAR](https://www.sec.gov/edgar/search/).

**4. Launch the app**

```bash
streamlit run streamlit_app.py
```

Then click **Run Ingestion Pipeline** in the sidebar.

---

## 🤖 LLM Models (Free Tier)

The app supports switching between three free-tier LLM providers at runtime via the sidebar dropdown.

| Provider | Model | Free Tier Details | API Key | Signup |
|---|---|---|---|---|
| **Gemini** (default) | Gemini 2.5 Flash | 1,500 req/day, 1M tokens/min | `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| **Groq** | Llama 3.3 70B | ~1,440 req/day, 60 RPM | `GROQ_API_KEY` | [Groq Console](https://console.groq.com/keys) |
| **OpenRouter** | Qwen 2.5 72B | Free/sponsored endpoints | `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/keys) |

Set `LLM_PROVIDER` in `.env` to the default, or switch models live in the UI.

---

## 📦 Tech Stack

| Component | Library |
|---|---|
| Distributed ingestion | PySpark 3.5 |
| PDF extraction | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (BAAI/bge-small-en-v1.5) |
| Vector DB | ChromaDB |
| LLM | Gemini 1.5 Flash / Llama 3.3 70B / Qwen 2.5 |
| UI | Streamlit |

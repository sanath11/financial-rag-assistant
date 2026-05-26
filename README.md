# 📈 Financial RAG Assistant

A **Retrieval-Augmented Generation (RAG)** pipeline for querying SEC filings (10-K/10-Q) using natural language. Built with **PySpark**, **ChromaDB**, **sentence-transformers**, and **Gemini 1.5 Flash** (free tier).

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
│  • Overlapping chunking     │  1000 chars, 100 overlap
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
┌─────────────────────────────┐
│  LLM Answer Generation      │  src/llm/gemini_svc.py
│  • Gemini 1.5 Flash (free)  │  1,500 req/day free
│  • RAG prompt template      │
│  • Streaming response       │
│  • Citation formatting      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Streamlit Web UI           │  streamlit_app.py
│  • Chat interface           │
│  • Ticker / year filters    │
│  • Source citations         │
│  • One-click ingestion      │
└─────────────────────────────┘
```

---

## 📁 Project Structure
```
financial-rag-assistant/
├── data/pdfs/
├── chroma_db/                    
├── src/
│   ├── ingestion/pyspark_processor.py
│   ├── embeddings/chroma_store.py
│   ├── llm/gemini_svc.py
│   ├── retrieval/search_engine.py
│   └── ui/components.py
├── requirements.txt
└── streamlit_app.py
```

---

## 🚀 Getting Started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add SEC filing PDFs** to `data/pdfs/` using this naming pattern:
```
nvda-20250126.pdf   (ticker-YYYYMMDD.pdf)
tsla-20241231.pdf
goog-20241231.pdf
```
Downloaded from [SEC EDGAR](https://www.sec.gov/edgar/search/).

**3. Launch the app**
```bash
streamlit run streamlit_app.py
```
Then click **Run Ingestion Pipeline** in the sidebar.

---

## 📦 Tech Stack

| Component | Library |
|---|---|
| Distributed ingestion | PySpark 3.5 |
| PDF extraction | PyMuPDF (fitz) |
| Embeddings | sentence-transformers |
| Vector DB | ChromaDB |
| LLM | Gemini 1.5 Flash |
| UI | Streamlit |

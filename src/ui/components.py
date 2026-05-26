"""
src/ui/components.py

Reusable Streamlit UI components for the Financial RAG Assistant.
Keeps streamlit_app.py clean by separating UI logic here.
"""

import streamlit as st


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(available_tickers: list[str], available_years: list[str]) -> dict:
    """
    Render the left sidebar with filters and database stats.

    Returns:
        dict: {selected_tickers, selected_years, top_k, show_sources}
    """
    st.sidebar.title("🔍 Search Filters")

    # ── Ticker filter ────────────────────────────────────────────────────────
    st.sidebar.subheader("Companies")
    if available_tickers:
        selected_tickers = st.sidebar.multiselect(
            "Filter by ticker",
            options=available_tickers,
            default=[],
            placeholder="All companies",
        )
    else:
        st.sidebar.warning("No data ingested yet.")
        selected_tickers = []

    # ── Year filter ───────────────────────────────────────────────────────────
    st.sidebar.subheader("Filing Year")
    if available_years:
        selected_years = st.sidebar.multiselect(
            "Filter by year",
            options=sorted(available_years, reverse=True),
            default=[],
            placeholder="All years",
        )
    else:
        selected_years = []

    # ── Advanced options ──────────────────────────────────────────────────────
    st.sidebar.subheader("⚙️ Settings")
    top_k = st.sidebar.slider(
        "Chunks to retrieve",
        min_value=2,
        max_value=10,
        value=6,
        help="More chunks = more context, but slower and more expensive",
    )

    show_sources = st.sidebar.toggle(
        "Show source passages",
        value=True,
        help="Display the exact document excerpts used to generate the answer",
    )

    # ── Model Selector ──────────────────────────────────────────────────────────
    st.sidebar.subheader("🤖 LLM Model")
    model_options = {
        "gemini": "Gemini 2.5 Flash (Google)",
        "llama": "Llama 3.3 70B (Groq)",
        "qwen": "Qwen 2.5 (OpenRouter)",
    }
    selected_model = st.sidebar.selectbox(
        "Select model",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0,
        help="Switch between free-tier LLM providers",
    )

    st.sidebar.divider()

    return {
        "selected_tickers": selected_tickers or None,
        "selected_years":   selected_years or None,
        "top_k":            top_k,
        "show_sources":     show_sources,
        "model":            selected_model,
    }


def render_ingestion_sidebar():
    """Sidebar section for running the ingestion pipeline."""
    st.sidebar.divider()
    st.sidebar.subheader("📥 Data Pipeline")

    if st.sidebar.button("⚡ Run Ingestion Pipeline", use_container_width=True):
        return True  # Signal to main app to run ingestion
    st.sidebar.caption("Extracts PDFs → chunks → stores embeddings in ChromaDB")
    return False


def render_db_stats(stats: dict):
    """Show ChromaDB stats in the sidebar."""
    st.sidebar.subheader("📊 Database")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Chunks", f"{stats.get('total_chunks', 0):,}")
    col2.metric("Tickers", len(stats.get("tickers", [])))
    if stats.get("tickers"):
        st.sidebar.caption(f"Companies: {', '.join(stats['tickers'])}")


# ── Header ────────────────────────────────────────────────────────────────────

def render_header():
    """Render the main page header and description."""
    st.title("📈 SEC Financial Assistant")
    st.caption(
        "Ask questions about SEC filings (10-K) for MAG 7 companies. "
        "Answers are grounded in actual filings with source citations."
    )


def render_example_questions(questions: list[str]) -> str | None:
    st.markdown("**💡 Suggested questions:**")
    cols = st.columns(2)
    for i, question in enumerate(questions):
        col = cols[i % 2]
        key = f"example_{i}_{hash(question) % 10000}"   # unique per question content
        if col.button(question, key=key, use_container_width=True):
            return question
    return None


# ── Answer Display ────────────────────────────────────────────────────────────

def render_answer(answer: str):
    """Render the LLM-generated answer."""
    st.markdown(answer)


def render_source_citations(chunks: list[dict]):
    """
    Render the retrieved source passages in an expandable section.
    Shows ticker, year, page, relevance score, and the actual text.
    """
    if not chunks:
        return

    with st.expander(f"📄 Source Documents ({len(chunks)} passages)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            score_pct = f"{chunk['score']:.0%}"
            header    = f"**[{i}] {chunk['source']}** — Relevance: {score_pct}"
            st.markdown(header)

            # Ticker badge + year badge
            col1, col2, col3 = st.columns([1, 1, 4])
            col1.markdown(f"`{chunk['ticker']}`")
            col2.markdown(f"`{chunk['year']}`")
            col3.markdown(f"Page {chunk['page']}")

            # Source text in a styled box
            st.markdown(
                f"""<div style="
                    background: #f8f9fa;
                    border-left: 3px solid #0066cc;
                    padding: 10px 14px;
                    border-radius: 4px;
                    font-size: 0.85em;
                    color: #333;
                    margin-bottom: 12px;
                ">{chunk['text'][:600]}{'...' if len(chunk['text']) > 600 else ''}</div>""",
                unsafe_allow_html=True,
            )


# ── Empty State ───────────────────────────────────────────────────────────────

def render_empty_state():
    """Show instructions when the database is empty."""
    st.info(
        "**No data found in the vector database.**\n\n"
        "To get started:\n"
        "1. Place your SEC filing PDFs in `data/pdfs/`\n"
        "2. Name them like: `nvda-20250126.pdf`, `tsla-20241231.pdf`\n"
        "3. Click **⚡ Run Ingestion Pipeline** in the sidebar\n\n"
        "The pipeline will extract text, generate embeddings, and store them in ChromaDB.",
        icon="📂",
    )


# ── Chat History ──────────────────────────────────────────────────────────────

def render_chat_message(role: str, content: str, sources: list[dict] | None = None):
    """Render a single chat message (user or assistant)."""
    with st.chat_message(role):
        st.markdown(content)
        if sources and role == "assistant":
            render_source_citations(sources)


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources_map" not in st.session_state:
        st.session_state.sources_map = {}
    if "suggested_questions_rendered" not in st.session_state:
        st.session_state["suggested_questions_rendered"] = False

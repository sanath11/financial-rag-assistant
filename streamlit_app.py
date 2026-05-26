"""
streamlit_app.py — Financial RAG Assistant
Main entry point. Run with: streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.embeddings.chroma_store import get_collection_stats
from src.retrieval.search_engine import retrieve_and_answer, get_available_tickers, get_available_years
from src.ui.components import (
    render_header, render_sidebar, render_ingestion_sidebar,
    render_db_stats, render_source_citations, render_empty_state,
    render_chat_message, init_chat_history,
)

# ── Session State ─────────────────────────────────────────────────────────────
init_chat_history()

# ── Cached Loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_db_stats():
    return get_collection_stats()

@st.cache_data(ttl=60)
def load_available_filters():
    return get_available_tickers(), get_available_years()

@st.cache_data(ttl=300)
def load_suggested_questions(tickers: tuple, years: tuple, model: str = "gemini") -> list[str]:
    from src.llm.llm_selector import generate_suggested_questions
    return generate_suggested_questions(list(tickers), list(years))

# ── Ingestion Pipeline ────────────────────────────────────────────────────────

def run_ingestion_pipeline():
    from src.ingestion.pyspark_processor import process_all_pdfs
    from src.embeddings.chroma_store import embed_and_store, clear_collection

    with st.status("Running ingestion pipeline...", expanded=True) as status:
        st.write("🔄 Starting PySpark session and scanning PDFs...")
        df_chunks = process_all_pdfs("data/pdfs/")

        if df_chunks is None or df_chunks.count() == 0:
            status.update(label="No PDFs found.", state="error")
            st.error("No PDFs found in `data/pdfs/`.")
            return

        st.write(f"✓ Extracted {df_chunks.count():,} chunks")
        st.write("🧹 Clearing old ChromaDB data...")
        clear_collection()
        st.write("⚡ Generating embeddings...")
        stored = embed_and_store(df_chunks)
        status.update(label=f"✅ Done — {stored:,} chunks stored!", state="complete")

    load_db_stats.clear()
    load_available_filters.clear()
    load_suggested_questions.clear()
    st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_header()

    available_tickers, available_years = load_available_filters()
    filters = render_sidebar(available_tickers, available_years)
    stats   = load_db_stats()
    render_db_stats(stats)

    # ── Set active LLM provider ───────────────────────────────────────────────
    if filters.get("model"):
        from src.llm.llm_selector import set_provider, get_provider
        set_provider(filters["model"])

    if render_ingestion_sidebar():
        run_ingestion_pipeline()
        return

    if stats["total_chunks"] == 0:
        render_empty_state()
        return

    # ── Chat History ──────────────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.messages):
        sources = st.session_state.sources_map.get(i) if msg["role"] == "assistant" else None
        render_chat_message(msg["role"], msg["content"], sources)

    # ── Chat Input ────────────────────────────────────────────────────────────
    user_query      = st.chat_input("Ask about SEC filings, earnings, risks, revenue...")
    question_to_run = user_query

    # ── Suggested Questions (inline buttons, no rerun needed) ─────────────────
    if not st.session_state.messages and not question_to_run:
        questions = load_suggested_questions(
            tickers=tuple(available_tickers),
            years=tuple(available_years),
            model=filters["model"],
        )
        st.markdown("**💡 Suggested questions:**")
        cols = st.columns(2)
        for i, q in enumerate(questions):
            if cols[i % 2].button(q, key=f"eq_{i}", use_container_width=True):
                question_to_run = q

    # ── RAG Pipeline ─────────────────────────────────────────────────────────
    if question_to_run:
        st.session_state.messages.append({"role": "user", "content": question_to_run})
        with st.chat_message("user"):
            st.markdown(question_to_run)

        with st.chat_message("assistant"):
            with st.spinner("Searching filings and generating answer..."):
                answer_stream, source_chunks = retrieve_and_answer(
                    question=question_to_run,
                    tickers=filters["selected_tickers"],
                    years=filters["selected_years"],
                    top_k=filters["top_k"],
                    stream=True,
                )
            answer_text = st.write_stream(answer_stream)
            if filters["show_sources"] and source_chunks:
                render_source_citations(source_chunks)

        msg_index = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
        st.session_state.sources_map[msg_index] = source_chunks

if __name__ == "__main__":
    main()
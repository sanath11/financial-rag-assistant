"""
streamlit_app.py — Financial RAG Assistant
Main entry point. Run with: streamlit run streamlit_app.py
"""

import sys
import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from src.embeddings.chroma_store import get_collection_stats
from src.retrieval.search_engine import (
    retrieve_and_answer,
    get_available_tickers,
    get_available_years,
)
from src.ui.components import (
    render_header,
    render_sidebar,
    render_ingestion_sidebar,
    render_db_stats,
    render_example_questions,
    render_answer,
    render_source_citations,
    render_empty_state,
    render_chat_message,
    init_chat_history,
)


# ── Session State ─────────────────────────────────────────────────────────────
init_chat_history()


# ── Cached Data Loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=60)   # Refresh every 60s so new ingestions appear
def load_db_stats():
    return get_collection_stats()


@st.cache_data(ttl=60)
def load_available_filters():
    return get_available_tickers(), get_available_years()


# ── Ingestion Pipeline Runner ─────────────────────────────────────────────────

def run_ingestion_pipeline():
    """Run the full PDF → ChromaDB ingestion pipeline with progress feedback."""
    from src.ingestion.pyspark_processor import process_all_pdfs
    from src.embeddings.chroma_store import embed_and_store, clear_collection

    with st.status("Running ingestion pipeline...", expanded=True) as status:
        st.write("🔄 Starting PySpark session and scanning PDFs...")
        df_chunks = process_all_pdfs("data/pdfs/")

        if df_chunks is None or df_chunks.count() == 0:
            status.update(label="No PDFs found.", state="error")
            st.error("No PDFs found in `data/pdfs/`. Add your SEC filing PDFs and retry.")
            return

        chunk_count = df_chunks.count()
        st.write(f"✓ Extracted {chunk_count:,} chunks from PDFs")

        st.write("🧹 Clearing old ChromaDB data...")
        clear_collection()

        st.write("⚡ Generating embeddings and storing in ChromaDB...")
        stored = embed_and_store(df_chunks)

        status.update(
            label=f"✅ Ingestion complete — {stored:,} chunks stored!",
            state="complete",
        )

    # Clear cache so sidebar stats refresh
    load_db_stats.clear()
    load_available_filters.clear()
    st.rerun()


# ── Main Layout ───────────────────────────────────────────────────────────────

def main():
    render_header()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    available_tickers, available_years = load_available_filters()
    filters = render_sidebar(available_tickers, available_years)
    stats   = load_db_stats()
    render_db_stats(stats)

    run_ingestion = render_ingestion_sidebar()
    if run_ingestion:
        run_ingestion_pipeline()
        return

    # ── Empty State ───────────────────────────────────────────────────────────
    if stats["total_chunks"] == 0:
        render_empty_state()
        return

    # ── Chat History ──────────────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.messages):
        sources = st.session_state.sources_map.get(i) if msg["role"] == "assistant" else None
        render_chat_message(msg["role"], msg["content"], sources)

    # ── Example Questions (shown when chat is empty) ──────────────────────────
    if not st.session_state.messages:
        clicked_question = render_example_questions()
        if clicked_question:
            st.session_state.messages.append({"role": "user", "content": clicked_question})
            st.rerun()

    # ── Chat Input ────────────────────────────────────────────────────────────
    user_query = st.chat_input(
        "Ask about SEC filings, earnings, risks, revenue... ",
        disabled=(stats["total_chunks"] == 0),
    )

    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate answer with streaming
        with st.chat_message("assistant"):
            with st.spinner("Searching filings and generating answer..."):
                answer_stream, source_chunks = retrieve_and_answer(
                    question=user_query,
                    tickers=filters["selected_tickers"],
                    years=filters["selected_years"],
                    top_k=filters["top_k"],
                    stream=True,
                )

            # Stream the response token by token
            answer_text = st.write_stream(answer_stream)

            # Show source citations below the answer
            if filters["show_sources"] and source_chunks:
                render_source_citations(source_chunks)

        # Save to session state
        msg_index = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
        st.session_state.sources_map[msg_index] = source_chunks


if __name__ == "__main__":
    main()

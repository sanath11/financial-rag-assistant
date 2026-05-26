"""
src/retrieval/search_engine.py

Semantic search over ChromaDB with optional metadata filtering.
Supports filtering by ticker, year, and page range.
"""

import sys
from sentence_transformers import SentenceTransformer

sys.path.append(".")
from src.embeddings.chroma_store import (
    get_chroma_client,
    get_or_create_collection,
    get_embedding_model,
    EMBEDDING_MODEL,
)

def expand_query(query: str) -> str:
    """Prepend context so embeddings align better with SEC filing language."""
    return f"financial SEC filing annual report: {query}"


# ── Core Search ───────────────────────────────────────────────────────────────
def semantic_search(
    query: str,
    tickers: list[str] | None = None,
    years: list[str] | None = None,
    top_k: int = 6,
) -> list[dict]:

    if not query or not query.strip():
        return []

    # Initialize model and collection
    model      = get_embedding_model(EMBEDDING_MODEL)
    client     = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        return []

    # Embed the query
    query_embedding = model.encode(
    [expand_query(query)],
    normalize_embeddings=True,
)[0].tolist()


    # Build metadata filter
    where_filter = _build_filter(tickers, years)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    chunks    = []
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, meta, distance in zip(docs, metadatas, distances):
        similarity = 1 - distance
        chunks.append({
            "text":      text,
            "source":    meta.get("source", "Unknown"),
            "ticker":    meta.get("ticker", ""),
            "year":      meta.get("year", ""),
            "page":      meta.get("page", 0),
            "file_name": meta.get("file_name", ""),
            "score":     round(similarity, 4),
        })

    return chunks


def _build_filter(
    tickers: list[str] | None,
    years: list[str] | None,
) -> dict | None:
    """
    Build a ChromaDB metadata filter.
    Supports combined ticker + year filtering.
    """
    conditions = []

    if tickers and len(tickers) == 1:
        conditions.append({"ticker": {"$eq": tickers[0]}})
    elif tickers and len(tickers) > 1:
        conditions.append({"ticker": {"$in": tickers}})

    if years and len(years) == 1:
        conditions.append({"year": {"$eq": years[0]}})
    elif years and len(years) > 1:
        conditions.append({"year": {"$in": years}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Context Formatting ────────────────────────────────────────────────────────

def format_context_for_llm(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block for the LLM prompt.
    Each chunk is labeled with its source for citation.
    """
    if not chunks:
        return "No relevant documents found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk['source']} (relevance: {chunk['score']:.0%})"
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def deduplicate_chunks(chunks: list[dict], min_similarity: float = 0.95) -> list[dict]:
    """
    Remove near-duplicate chunks (same page, very similar text).
    Happens when chunk overlap creates almost-identical results.
    """
    seen_keys = set()
    unique    = []
    for chunk in chunks:
        key = (chunk["file_name"], chunk["page"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(chunk)
    return unique


# ── Available Filters (for UI dropdowns) ─────────────────────────────────────

def get_available_tickers() -> list[str]:
    """Return all tickers currently stored in ChromaDB."""
    from src.embeddings.chroma_store import get_collection_stats
    return get_collection_stats().get("tickers", [])


def get_available_years() -> list[str]:
    """Return all years currently stored in ChromaDB."""
    from src.embeddings.chroma_store import get_collection_stats
    return get_collection_stats().get("years", [])


# ── Full RAG Pipeline Call ────────────────────────────────────────────────────

def retrieve_and_answer(
    question: str,
    tickers: list[str] | None = None,
    years: list[str] | None = None,
    top_k: int = 6,
    stream: bool = False,
):
    """
    Full RAG pipeline: retrieve relevant chunks → generate LLM answer.

    Args:
        question: User's question
        tickers:  Filter by tickers
        years:    Filter by years
        top_k:    Number of chunks to retrieve
        stream:   If True, returns a generator for streaming

    Returns:
        (answer_str, source_chunks) or (generator, source_chunks) if stream=True
    """
    from src.llm.llm_selector import generate_answer, generate_answer_stream

    # Retrieve
    chunks = semantic_search(question, tickers=tickers, years=years, top_k=top_k * 2)
    chunks = [c for c in chunks if c["score"] >= 0.55]   # Drop low-quality chunks
    chunks = chunks[:top_k]
    chunks = deduplicate_chunks(chunks)

    if not chunks:
        no_data_msg = (
            "No relevant documents found for your query. "
            "Try adjusting your filters or re-phrasing your question."
        )
        return no_data_msg, []

    # Generate
    if stream:
        answer_gen = generate_answer_stream(question, chunks, tickers=tickers)
        return answer_gen, chunks
    else:
        answer = generate_answer(question, chunks, tickers=tickers)
        return answer, chunks


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_query = "What are NVIDIA's main revenue drivers?"
    print(f"Query: {test_query}\n")

    chunks = semantic_search(test_query, tickers=["NVDA"], top_k=3)
    if chunks:
        print(f"Retrieved {len(chunks)} chunks:\n")
        for c in chunks:
            print(f"  [{c['score']:.0%}] {c['source']}")
            print(f"  {c['text'][:200]}...\n")
    else:
        print("No results found. Have you run the ingestion pipeline?")

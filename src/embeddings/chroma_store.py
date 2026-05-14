"""
src/embeddings/chroma_store.py

Converts Spark DataFrame chunks → embeddings → ChromaDB vector store.

Flow:
    Spark DataFrame (file_name, ticker, year, page, chunk_text)
        ↓ collect to driver
        ↓ batch embed with sentence-transformers
        ↓ upsert into ChromaDB with metadata
"""

import hashlib
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH      = "chroma_db"
COLLECTION_NAME     = "financial_sec_filings"
EMBEDDING_MODEL     = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 64


# ── ChromaDB Client ───────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client pointing to local chroma_db/."""
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    """Get or create the financial filings collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # Cosine similarity for text
    )


# ── Embedding Model ───────────────────────────────────────────────────────────

_model_cache = {}

def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load model once and cache it (avoids reloading on every call)."""
    if model_name not in _model_cache:
        print(f"Loading embedding model: {model_name}...")
        _model_cache[model_name] = SentenceTransformer(model_name)
        print("Model loaded ✓")
    return _model_cache[model_name]


# ── Chunk ID ──────────────────────────────────────────────────────────────────

def make_chunk_id(file_name: str, page: int, chunk_index: int) -> str:
    """
    Deterministic ID so re-running the pipeline doesn't create duplicates.
    ChromaDB upsert will overwrite existing IDs.
    """
    raw = f"{file_name}::page{page}::chunk{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Main Ingestion Function ───────────────────────────────────────────────────

def embed_and_store(spark_df, batch_size: int = EMBEDDING_BATCH_SIZE) -> int:
    """
    Take a Spark DataFrame of chunks and store embeddings in ChromaDB.

    Args:
        spark_df: DataFrame with columns (file_name, ticker, year, page, chunk_text)
        batch_size: How many chunks to embed at once

    Returns:
        Total number of chunks stored
    """
    # Collect from Spark to driver (embedding happens locally)
    print("Collecting chunks from Spark...")
    rows = spark_df.collect()
    print(f"Collected {len(rows)} chunks")

    if not rows:
        print("No chunks to embed.")
        return 0

    # Load model and ChromaDB
    model      = get_embedding_model()
    client     = get_chroma_client()
    collection = get_or_create_collection(client)

    # Process in batches
    total_stored = 0
    chunk_counter = {}   # Track chunk index per page for ID generation

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]

        texts     = [row.chunk_text for row in batch]
        metadatas = []
        ids       = []

        for row in batch:
            page_key = f"{row.file_name}::page{row.page}"
            chunk_counter[page_key] = chunk_counter.get(page_key, 0) + 1
            chunk_idx = chunk_counter[page_key]

            ids.append(make_chunk_id(row.file_name, row.page, chunk_idx))
            metadatas.append({
                "file_name": row.file_name,
                "ticker":    row.ticker,
                "year":      row.year,
                "page":      int(row.page),
                "chunk_idx": chunk_idx,
                # Source label shown in citations
                "source":    f"{row.ticker} ({row.year}) — Page {row.page}",
            })

        # Generate embeddings for the batch
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Required for cosine similarity
        ).tolist()

        # Upsert into ChromaDB (safe to re-run — won't duplicate)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_stored += len(batch)
        pct = (total_stored / len(rows)) * 100
        print(f"  Embedded {total_stored}/{len(rows)} chunks ({pct:.0f}%)", end="\r")

    print(f"\n✓ Stored {total_stored} chunks in ChromaDB collection '{COLLECTION_NAME}'")
    return total_stored


# ── Utility: Collection Stats ─────────────────────────────────────────────────

def get_collection_stats() -> dict:
    """Return summary of what's currently stored in ChromaDB."""
    client     = get_chroma_client()
    collection = get_or_create_collection(client)
    count      = collection.count()

    stats = {"total_chunks": count, "tickers": [], "years": []}

    if count > 0:
        # Sample metadata to find tickers/years
        sample = collection.get(limit=min(count, 1000), include=["metadatas"])
        tickers = sorted(set(m["ticker"] for m in sample["metadatas"]))
        years   = sorted(set(m["year"]   for m in sample["metadatas"]))
        stats["tickers"] = tickers
        stats["years"]   = years

    return stats


def clear_collection():
    """Delete and recreate the collection (for re-ingestion)."""
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    get_or_create_collection(client)
    print(f"Recreated empty collection '{COLLECTION_NAME}'")


# ── CLI: Run full ingestion pipeline ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.ingestion.pyspark_processor import process_all_pdfs

    pdf_dir = "data/pdfs/"
    print(f"Running ingestion pipeline on {pdf_dir}")

    df_chunks = process_all_pdfs(pdf_dir)
    if df_chunks is not None:
        embed_and_store(df_chunks)

        # Print final stats
        stats = get_collection_stats()
        print(f"\nChromaDB Summary:")
        print(f"  Total chunks : {stats['total_chunks']}")
        print(f"  Tickers      : {stats['tickers']}")
        print(f"  Years        : {stats['years']}")

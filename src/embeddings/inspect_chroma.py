import sys
sys.path.append(".")
from src.embeddings.chroma_store import get_chroma_client, COLLECTION_NAME

client     = get_chroma_client()
collection = client.get_collection(COLLECTION_NAME)

# ── Stats ──────────────────────────────────────────────────────
print(f"Total chunks : {collection.count()}")

# ── Sample 5 records ───────────────────────────────────────────
results = collection.get(limit=5, include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n[{i+1}] {meta}")
    print(f"     {doc[:200]}...")

# ── Filter by ticker ───────────────────────────────────────────
nvda = collection.get(
    where={"ticker": {"$eq": "NVDA"}},
    limit=3,
    include=["documents", "metadatas"],
)
print(f"\nNVDA chunks: {len(nvda['documents'])}")
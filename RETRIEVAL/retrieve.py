"""
retrieval/retrieve.py
---------------------
Semantic search over ChromaDB.

Improvement 2: Upgraded from all-MiniLM-L6-v2 to BAAI/bge-large-en-v1.5

WHY bge-large over MiniLM:
  - MiniLM-L6-v2: 384 dimensions, trained on general web text
  - bge-large-en-v1.5: 1024 dimensions, trained with MSMARCO + legal-adjacent
    corpora, consistently outperforms MiniLM on BEIR retrieval benchmarks by
    5-8 points. Especially better on domain-specific long-form text like contracts.

TRADEOFF: bge-large is ~1.3GB vs MiniLM's ~90MB. Embedding is slower on CPU
(~2x). Acceptable for this prototype — still fast enough for demo.

INTENTIONALLY NOT ADDED:
- legal-bert: Fine-tuned for classification, not retrieval. Poor on BEIR.
- Instructor-XL: Strong but 5GB, slow on CPU, overkill for 50 contracts.
- Hybrid BM25+Dense: Meaningful improvement but adds rank fusion complexity
  and a separate BM25 index. Documented as next step.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "legal_docs"

# Upgraded embedding model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# bge models need a query prefix for retrieval (per BAAI's instructions)
# This is important — without it, bge underperforms significantly
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_TOP_K = 5

_model = None
_client = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    rewrite: bool = False,
) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve the top-K most relevant chunks from ChromaDB.

    Args:
        query: User's question (plain English or legal)
        top_k: Number of chunks to retrieve
        rewrite: If True, rewrite query to legal terminology before embedding

    Returns:
        List of dicts with keys: text, title, score, doc_idx, chunk_idx,
        query_used, original_query
    """
    original_query = query

    if rewrite:
        from retrieval.query_rewriter import rewrite_query
        query = rewrite_query(query)
        logger.info(f"Query rewritten: '{original_query[:50]}' → '{query[:50]}'")

    model = _get_model()
    collection = _get_collection()

    # bge requires query prefix for retrieval tasks
    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = model.encode([prefixed_query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for doc, meta, dist in zip(docs, metas, distances):
        retrieved.append({
            "text": doc,
            "title": meta.get("title", "unknown"),
            "score": round(1 - dist, 4),
            "doc_idx": meta.get("doc_idx"),
            "chunk_idx": meta.get("chunk_idx"),
            "query_used": query,
            "original_query": original_query,
        })

    return retrieved


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into context string for LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i} — {chunk['title']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    query = "What are the termination clauses in this contract?"
    results = retrieve(query, top_k=3, rewrite=True)
    for r in results:
        print(f"\n[{r['title']}] (score: {r['score']})")
        print(f"Query used: {r['query_used']}")
        print(r["text"][:300], "...")

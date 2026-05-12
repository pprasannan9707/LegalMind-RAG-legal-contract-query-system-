"""
eval/failure_modes.py
---------------------
Documents and tests 2 failure modes:

  FAILURE MODE 1: Chunk size too large (512 words)
    -> Retrieves overly broad chunks that dilute the relevant legal clause
    -> FIX: Reduce chunk size to 128 words for clause-level granularity
    -> Shows before/after eval scores

  FAILURE MODE 2: Semantic mismatch (vocabulary gap)
    -> User says "firing" but contract says "termination"
    -> Retrieval fails because embeddings don't bridge the gap well enough
    -> Documented with score evidence
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from eval.evaluator import run_eval_suite, compare_before_after


def demonstrate_semantic_mismatch():
    """
    FAILURE MODE 2: Show how vocabulary gap hurts retrieval.
    Query uses colloquial terms; documents use formal legal language.
    """
    print("\n" + "="*60)
    print("FAILURE MODE 2: Semantic Mismatch (Vocabulary Gap)")
    print("="*60)

    mismatch_pairs = [
        ("firing an employee", "termination of employment"),
        ("getting out of the deal", "contract cancellation or rescission"),
        ("not telling others about the business", "confidentiality and non-disclosure"),
        ("owning the invention", "intellectual property assignment"),
        ("suing someone", "litigation and legal proceedings"),
    ]

    from sentence_transformers import SentenceTransformer, util
    from eval.evaluator import EMBED_MODEL

    model = SentenceTransformer(EMBED_MODEL)    

    print("\nSemantic similarity between colloquial and legal terms:\n")
    print(f"{'Colloquial Query':<42} {'Legal Term':<42} {'Similarity':>10}")
    print("-" * 96)

    for colloquial, legal in mismatch_pairs:
        e1 = model.encode(colloquial, convert_to_tensor=True)
        e2 = model.encode(legal, convert_to_tensor=True)
        sim = float(util.cos_sim(e1, e2))
        flag = "⚠️  LOW" if sim < 0.6 else "✅ OK"
        print(f"{colloquial:<42} {legal:<42} {sim:>8.3f}  {flag}")

    print("""
Analysis:
  - Similarities below 0.6 indicate a vocabulary gap
  - When users phrase queries colloquially, retrieval may miss
    the correct legal clauses because the embedding space
    doesn't fully bridge formal vs informal legal language

Potential Fixes (future work):
  - Query expansion: use LLM to rewrite query in legal terminology
  - HyDE (Hypothetical Document Embeddings): generate a fake legal
    passage matching the query, embed that instead
  - Fine-tune embeddings on legal corpora (legal-bert)
""")


def rebuild_chroma_with_chunk_size(chunk_size: int):
    """
    Re-chunk and re-embed the documents already stored in ChromaDB
    using a different chunk size. Reads raw contexts from existing collection.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer, util
    from eval.evaluator import EMBED_MODEL
    from tqdm import tqdm
    from ingestion.ingest import CHROMA_DIR, COLLECTION_NAME, chunk_text

    print(f"\nRebuilding ChromaDB with chunk_size={chunk_size}...")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Read all existing documents from the current collection
    try:
        existing = client.get_collection(COLLECTION_NAME)
        all_data = existing.get(include=["documents", "metadatas"])
        raw_docs = all_data["documents"]
        raw_metas = all_data["metadatas"]
    except Exception as e:
        print(f"Could not read existing collection: {e}")
        return None

    if not raw_docs:
        print("No documents found in existing ChromaDB. Run --step ingest first.")
        return None

    # Reconstruct unique full contexts by joining chunks per doc_idx
    from collections import defaultdict
    doc_chunks = defaultdict(list)
    doc_titles = {}
    for doc, meta in zip(raw_docs, raw_metas):
        idx = meta.get("doc_idx", 0)
        chunk_idx = meta.get("chunk_idx", 0)
        doc_chunks[idx].append((chunk_idx, doc))
        doc_titles[idx] = meta.get("title", "unknown")

    # Rebuild full contexts
    full_docs = []
    for idx in sorted(doc_chunks.keys()):
        sorted_chunks = sorted(doc_chunks[idx], key=lambda x: x[0])
        full_context = " ".join(c[1] for c in sorted_chunks)
        full_docs.append({"title": doc_titles[idx], "context": full_context})

    print(f"Reconstructed {len(full_docs)} documents from existing ChromaDB")

    # Delete old collection and rebuild with new chunk size
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    model = SentenceTransformer(EMBED_MODEL)   
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks, all_ids, all_metadatas = [], [], []
    for doc_idx, doc in enumerate(tqdm(full_docs, desc="Re-chunking")):
        chunks = chunk_text(doc["context"], chunk_size=chunk_size)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"doc{doc_idx}_chunk{chunk_idx}")
            all_metadatas.append({
                "title": doc["title"],
                "doc_idx": doc_idx,
                "chunk_idx": chunk_idx,
                "chunk_size_used": chunk_size,
            })

    print(f"Embedding {len(all_chunks)} chunks...")
    BATCH = 256
    all_embeddings = []
    for i in tqdm(range(0, len(all_chunks), BATCH), desc="Embedding"):
        batch = all_chunks[i: i + BATCH]
        embs = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embs)

    UPSERT_BATCH = 500
    for i in range(0, len(all_chunks), UPSERT_BATCH):
        collection.upsert(
            ids=all_ids[i: i + UPSERT_BATCH],
            documents=all_chunks[i: i + UPSERT_BATCH],
            embeddings=all_embeddings[i: i + UPSERT_BATCH],
            metadatas=all_metadatas[i: i + UPSERT_BATCH],
        )

    print(f"Stored {len(all_chunks)} chunks (chunk_size={chunk_size}) in ChromaDB")
    return collection


def run_failure_mode_1_fix(eval_pairs, max_pairs=20):
    """
    FAILURE MODE 1: Large chunks (512 words) -> imprecise retrieval
    FIX: Small chunks (128 words) -> clause-level precision
    """
    print("\n" + "="*60)
    print("FAILURE MODE 1: Chunk Size Too Large")
    print("="*60)
    print("""
Problem:
  Chunk size of 512 words retrieves entire contract sections.
  When a question asks about a specific clause (e.g. termination
  notice period), the retrieved chunk contains the clause PLUS
  surrounding irrelevant text. This dilutes the context and
  confuses the LLM, reducing answer precision.

Fix:
  Reduce chunk size from 512 -> 128 words.
  Each chunk now maps to a single clause or paragraph,
  so retrieval is more targeted and context is cleaner.
""")

    import retrieval.retrieve as rr

    # ── BEFORE: use existing chunk_size=512 collection ────────────────────────
    print("Running BEFORE eval (chunk_size=512, existing collection)...")
    rr._collection = None
    rr._client = None
    df_before = run_eval_suite(
        eval_pairs,
        chunk_size=512,
        output_prefix="eval/results",
        max_pairs=max_pairs,
    )

    # ── FIX: rebuild with chunk_size=128 ─────────────────────────────────────
    rebuild_chroma_with_chunk_size(chunk_size=128)

    # Reset singleton so retriever picks up new collection
    rr._collection = None
    rr._client = None

    # ── AFTER: eval with chunk_size=128 ──────────────────────────────────────
    print("\nRunning AFTER eval (chunk_size=128)...")
    df_after = run_eval_suite(
        eval_pairs,
        chunk_size=128,
        output_prefix="eval/results",
        max_pairs=max_pairs,
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    comparison = compare_before_after(
        df_before, df_after,
        "chunk=512 (before)",
        "chunk=128 (after)"
    )
    print("\n" + "="*60)
    print("  BEFORE vs AFTER COMPARISON")
    print("="*60)
    print(comparison.to_string(index=False))
    comparison.to_csv("eval/before_after_comparison.csv", index=False)
    print("\nSaved to eval/before_after_comparison.csv")

    return df_before, df_after, comparison


if __name__ == "__main__":
    with open("data/eval_pairs.json") as f:
        pairs = json.load(f)

    demonstrate_semantic_mismatch()
    run_failure_mode_1_fix(pairs, max_pairs=20)

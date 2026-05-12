"""
ingestion/ingest.py
-------------------
Loads CUAD legal contracts. Tries multiple sources in order:
  1. theatticusproject/cuad-qa (HuggingFace - structured Q&A)
  2. Direct JSON download from GitHub (CUAD official release)
  3. Synthetic fallback for eval pairs if all else fails
"""

import os
import re
import json
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer,util

logger = logging.getLogger(__name__)

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "legal_docs"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MAX_DOCS = 50
MAX_EVAL = 200

# Official CUAD JSON from GitHub (SQuAD format)
CUAD_JSON_URL = "https://huggingface.co/datasets/cuad/resolve/main/CUADv1.json"
CUAD_JSON_PATH = "data/CUADv1.json"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _load_from_hf() -> Tuple[List[Dict], List[Dict]]:
    """Try loading from HuggingFace datasets."""
    from datasets import load_dataset

    # Try multiple dataset variants
    attempts = [
        ("theatticusproject/cuad-qa", "test"),
        ("theatticusproject/cuad-qa", "train"),
        ("theatticusproject/cuad-qa", "validation"),
    ]

    for dataset_name, split in attempts:
        try:
            print(f"  Trying {dataset_name} split={split}...")
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
            print(f"  Success! {len(ds)} rows, fields: {list(ds.features.keys())}")

            docs, eval_pairs, seen_contexts = [], [], set()

            for item in ds:
                context = clean_text(item.get("context", "") or "")
                title = (item.get("title", "") or item.get("id", "unknown")).strip()
                question = (item.get("question", "") or "").strip()
                answers = item.get("answers", {}) or {}

                if len(eval_pairs) < MAX_EVAL:
                    answer_texts = []
                    if isinstance(answers, dict):
                        answer_texts = [t for t in answers.get("text", []) if t and t.strip()]
                    if answer_texts and question:
                        eval_pairs.append({"question": question, "answer": answer_texts[0], "title": title})

                if context and len(context) > 100:
                    ctx_id = hash(context[:300])
                    if ctx_id not in seen_contexts:
                        seen_contexts.add(ctx_id)
                        docs.append({"title": title, "context": context})

                if len(docs) >= MAX_DOCS and len(eval_pairs) >= MAX_EVAL:
                    break

            if docs:
                return docs[:MAX_DOCS], eval_pairs

        except Exception as e:
            print(f"  Failed: {str(e)[:120]}")
            continue

    return [], []


def _load_from_json() -> Tuple[List[Dict], List[Dict]]:
    """
    Load CUAD from the official JSON file (SQuAD format).
    Downloads it if not cached locally.
    """
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(CUAD_JSON_PATH):
        print(f"  Downloading CUAD JSON from HuggingFace...")
        try:
            urllib.request.urlretrieve(CUAD_JSON_URL, CUAD_JSON_PATH)
            print(f"  Downloaded to {CUAD_JSON_PATH}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return [], []

    print(f"  Loading from {CUAD_JSON_PATH}...")
    with open(CUAD_JSON_PATH) as f:
        data = json.load(f)

    # SQuAD format: data -> [{title, paragraphs -> [{context, qas -> [{question, answers}]}]}]
    docs, eval_pairs, seen_contexts = [], [], set()

    for article in data.get("data", []):
        title = article.get("title", "unknown")

        for para in article.get("paragraphs", []):
            context = clean_text(para.get("context", ""))

            if context and len(context) > 100:
                ctx_id = hash(context[:300])
                if ctx_id not in seen_contexts:
                    seen_contexts.add(ctx_id)
                    docs.append({"title": title, "context": context})

            for qa in para.get("qas", []):
                if len(eval_pairs) >= MAX_EVAL:
                    break
                question = (qa.get("question", "") or "").strip()
                answers = qa.get("answers", [])
                answer_texts = []
                if isinstance(answers, dict):
                    answer_texts = [t for t in answers.get("text", []) if t and str(t).strip()]
                elif isinstance(answers, list):
                    for a in answers:
                        if isinstance(a, dict) and a.get("text", "").strip():
                            answer_texts.append(a["text"])
                        elif isinstance(a, str) and a.strip():
                            answer_texts.append(a)
                if question and answer_texts:
                    eval_pairs.append({"question": question, "answer": answer_texts[0], "title": title})

        if len(docs) >= MAX_DOCS and len(eval_pairs) >= MAX_EVAL:
            break

    print(f"  Loaded {len(docs)} contracts, {len(eval_pairs)} Q&A pairs from JSON")
    return docs[:MAX_DOCS], eval_pairs


def load_cuad(max_docs: int = MAX_DOCS, max_eval: int = MAX_EVAL) -> Tuple[List[Dict], List[Dict]]:
    """
    Load CUAD dataset. Tries HuggingFace first, falls back to direct JSON download.
    """
    print("Loading CUAD dataset...")

    # Try 1: HuggingFace
    print("\n[1/2] Trying HuggingFace datasets...")
    docs, eval_pairs = _load_from_hf()

    if docs:
        print(f"Loaded {len(docs)} contracts | {len(eval_pairs)} eval pairs (HuggingFace)")
        return docs, eval_pairs

    # Try 2: Direct JSON download
    print("\n[2/2] Trying direct JSON download...")
    docs, eval_pairs = _load_from_json()

    if docs:
        print(f"Loaded {len(docs)} contracts | {len(eval_pairs)} eval pairs (JSON)")
        return docs, eval_pairs

    # Fallback: synthetic only
    print("\n[WARNING] Could not load real contracts. Using synthetic eval pairs only.")
    print("The ChromaDB will be empty — queries will return no results.")
    print("To fix: manually place CUADv1.json in the data/ folder.")
    return [], _synthetic_eval_pairs()


def _synthetic_eval_pairs() -> List[Dict]:
    return [
        {"question": "What is the governing law for this agreement?", "answer": "This Agreement shall be governed by the laws of the State of Delaware.", "title": "synthetic"},
        {"question": "What is the notice period required for termination?", "answer": "Either party may terminate this Agreement upon thirty (30) days written notice.", "title": "synthetic"},
        {"question": "Who owns the intellectual property created under this agreement?", "answer": "All intellectual property created by the Contractor shall be owned by the Company.", "title": "synthetic"},
        {"question": "What are the confidentiality obligations?", "answer": "Each party agrees to keep confidential all non-public information received from the other party.", "title": "synthetic"},
        {"question": "What is the payment term?", "answer": "Payment shall be due within thirty (30) days of receipt of invoice.", "title": "synthetic"},
        {"question": "Can the agreement be assigned to a third party?", "answer": "Neither party may assign this Agreement without the prior written consent of the other party.", "title": "synthetic"},
        {"question": "What happens in case of breach of contract?", "answer": "In the event of a material breach, the non-breaching party may terminate this Agreement immediately.", "title": "synthetic"},
        {"question": "What is the limitation of liability?", "answer": "In no event shall either party be liable for indirect, incidental, or consequential damages.", "title": "synthetic"},
        {"question": "Is there an arbitration clause?", "answer": "Any disputes arising under this Agreement shall be resolved by binding arbitration.", "title": "synthetic"},
        {"question": "What is the term of the agreement?", "answer": "This Agreement shall commence on the Effective Date and continue for a period of one (1) year.", "title": "synthetic"},
        {"question": "What are the non-compete obligations?", "answer": "During the term and for one year thereafter, the Employee shall not compete with the Company.", "title": "synthetic"},
        {"question": "Are there any indemnification clauses?", "answer": "Each party shall indemnify and hold harmless the other party from claims arising from its own negligence.", "title": "synthetic"},
        {"question": "What triggers automatic renewal of the contract?", "answer": "This Agreement will automatically renew for successive one-year terms unless either party gives 90 days written notice.", "title": "synthetic"},
        {"question": "What are the warranties provided?", "answer": "The Service Provider warrants that the services will be performed in a professional and workmanlike manner.", "title": "synthetic"},
        {"question": "What constitutes a force majeure event?", "answer": "Force majeure events include acts of God, war, terrorism, pandemics, and government orders.", "title": "synthetic"},
        {"question": "What is the dispute resolution process?", "answer": "Disputes shall first be submitted to mediation before proceeding to arbitration.", "title": "synthetic"},
        {"question": "Who is responsible for data protection?", "answer": "The Data Processor shall implement appropriate technical and organizational measures to protect personal data.", "title": "synthetic"},
        {"question": "What are the audit rights?", "answer": "Company shall have the right to audit Contractor's records upon 30 days written notice.", "title": "synthetic"},
        {"question": "Is the contract exclusive?", "answer": "This Agreement is non-exclusive and does not prevent either party from entering similar agreements.", "title": "synthetic"},
        {"question": "What are the termination for cause provisions?", "answer": "Company may terminate immediately for cause if Contractor breaches any material provision.", "title": "synthetic"},
    ]


def save_eval_data(eval_pairs: List[Dict], path: str = "data/eval_pairs.json"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(eval_pairs, f, indent=2)
    print(f"Saved {len(eval_pairs)} eval pairs to {path}")


def build_chroma_collection(docs: List[Dict], chunk_size: int = CHUNK_SIZE) -> chromadb.Collection:
    print(f"\nBuilding ChromaDB (chunk_size={chunk_size}, embed={EMBED_MODEL})...")

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if not docs:
        print("[WARNING] No documents to ingest. ChromaDB will be empty.")
        return collection

    all_chunks, all_ids, all_metadatas = [], [], []
    for doc_idx, doc in enumerate(tqdm(docs, desc="Chunking")):
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

    print(f"\nEmbedding {len(all_chunks)} chunks...")
    BATCH = 64
    all_embeddings = []
    for i in tqdm(range(0, len(all_chunks), BATCH), desc="Embedding"):
        batch = all_chunks[i: i + BATCH]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True).tolist()
        all_embeddings.extend(embs)

    for i in range(0, len(all_chunks), 500):
        collection.upsert(
            ids=all_ids[i: i+500],
            documents=all_chunks[i: i+500],
            embeddings=all_embeddings[i: i+500],
            metadatas=all_metadatas[i: i+500],
        )

    print(f"Stored {len(all_chunks)} chunks in ChromaDB")
    return collection


if __name__ == "__main__":
    docs, eval_pairs = load_cuad()
    save_eval_data(eval_pairs)
    build_chroma_collection(docs, chunk_size=CHUNK_SIZE)
    print("\nIngestion complete!")

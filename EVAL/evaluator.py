"""
eval/evaluator.py
-----------------
Custom evaluation framework for LegalMind RAG.

Improvement 3: Better Evaluation Metrics

WHY: The original eval had only retrieval precision with a hard binary threshold.
This is too coarse — it doesn't tell you HOW relevant the chunks are, or whether
the right chunk appeared at rank 1 vs rank 5.

Added metrics:
  - Precision@K  : fraction of top-K chunks above relevance threshold
  - Recall@K     : fraction of ground truth covered by top-K chunks  
  - MRR          : Mean Reciprocal Rank — rewards finding the right chunk early
  - Faithfulness : LLM-as-judge (unchanged, already working well)
  - Answer Relevance : semantic similarity query <-> answer (unchanged)
  - Hallucination Rate : 1 - faithfulness (derived)

INTENTIONALLY NOT ADDED:
  - RAGAS: External dependency, requires OpenAI by default, overkill for 20 pairs
  - ROUGE/BLEU: Designed for summarization, not QA — meaningless here
  - Human eval pipeline: Not feasible in 2-day trial scope
"""

import os
import json
import re
import time
import logging
import pandas as pd
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


# ── Metric 1: Precision@K ─────────────────────────────────────────────────────
def precision_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_answer: str,
    k: int = 3,
    threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Precision@K: Of the top-K retrieved chunks, what fraction are relevant
    to the ground truth answer?

    Relevance = cosine similarity >= threshold between chunk and ground truth.
    Threshold of 0.20 chosen empirically for short synthetic answers vs
    longer contract chunks. Would be higher (0.4+) with real CUAD Q&A pairs.
    """
    model = _get_embed_model()
    gt_emb = model.encode(ground_truth_answer, convert_to_tensor=True)

    top_k_chunks = retrieved_chunks[:k]
    scores = []
    for chunk in top_k_chunks:
        chunk_emb = model.encode(chunk["text"], convert_to_tensor=True)
        sim = float(util.cos_sim(gt_emb, chunk_emb))
        scores.append(sim)

    relevant_count = sum(1 for s in scores if s >= threshold)
    precision = relevant_count / len(scores) if scores else 0.0

    return {
        "precision_at_k": round(precision, 4),
        "k": k,
        "chunk_scores": [round(s, 4) for s in scores],
        "relevant_chunks": relevant_count,
        "max_chunk_score": round(max(scores), 4) if scores else 0.0,
    }


# ── Metric 2: Recall@K ───────────────────────────────────────────────────────
def recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_answer: str,
    k: int = 3,
    threshold: float = 0.20,
) -> float:
    """
    Recall@K: Does at least one of the top-K chunks contain the answer?

    Binary for single ground truth: 1.0 if any chunk is relevant, 0.0 if not.
    This is the most important retrieval metric for QA systems — you can't
    generate a correct answer if the right chunk isn't retrieved at all.
    """
    model = _get_embed_model()
    gt_emb = model.encode(ground_truth_answer, convert_to_tensor=True)

    for chunk in retrieved_chunks[:k]:
        chunk_emb = model.encode(chunk["text"], convert_to_tensor=True)
        sim = float(util.cos_sim(gt_emb, chunk_emb))
        if sim >= threshold:
            return 1.0
    return 0.0


# ── Metric 3: MRR (Mean Reciprocal Rank) ─────────────────────────────────────
def reciprocal_rank(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_answer: str,
    threshold: float = 0.20,
) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant chunk.
    - First chunk is relevant → RR = 1.0 (perfect)
    - Second chunk is relevant → RR = 0.5
    - Fifth chunk is relevant → RR = 0.2
    - No relevant chunk → RR = 0.0

    MRR averaged over all queries rewards systems that surface relevant
    content early, not just eventually. Critical for legal RAG where
    users read top-1 or top-2 results.
    """
    model = _get_embed_model()
    gt_emb = model.encode(ground_truth_answer, convert_to_tensor=True)

    for rank, chunk in enumerate(retrieved_chunks, 1):
        chunk_emb = model.encode(chunk["text"], convert_to_tensor=True)
        sim = float(util.cos_sim(gt_emb, chunk_emb))
        if sim >= threshold:
            return round(1.0 / rank, 4)
    return 0.0


# ── Metric 4: Answer Faithfulness (LLM-as-judge) ─────────────────────────────
FAITHFULNESS_PROMPT = """You are a strict legal RAG evaluator.

Your job is to determine whether the AI Answer is fully supported by the provided Context.

IMPORTANT RULES:
- Evaluate ONLY based on the Context.
- Do NOT use outside knowledge.
- Do NOT assume missing information.
- Do NOT allow logical inference unless explicitly stated.
- Every factual or legal claim in the AI Answer must be directly grounded in the Context.
- If the answer adds legal interpretation, assumptions, summaries, or conclusions not explicitly written in the Context, mark them as unsupported.
- Partial support is NOT full support.
- Legal wording must be treated strictly and precisely.

Context:
---------------------
{context}
---------------------

Question:
{question}

AI Answer:
---------------------
{answer}
---------------------

Evaluation Instructions:
1. Identify every factual, contractual, or legal claim in the AI Answer.
2. Check whether each claim is explicitly supported by the Context.
3. Penalize:
   - hallucinated legal terms
   - inferred obligations
   - unsupported conclusions
   - invented clause meanings
   - generalized summaries not present in the text
4. If even part of a claim is unsupported, include it in unsupported_claims.

Scoring Guidelines:
- 1.0 = every claim fully supported
- 0.8 = mostly supported with very minor unsupported wording
- 0.5 = partially grounded but contains notable unsupported claims
- 0.2 = largely unsupported or speculative
- 0.0 = fabricated answer unrelated to context

Respond ONLY in valid JSON.

Expected JSON format:
{
  "faithfulness_score": <float>,
  "reasoning": "<brief explanation>",
  "supported_claims": [
    "<claim>"
  ],
  "unsupported_claims": [
    "<claim>"
  ]
}
"""


def answer_faithfulness(query: str, answer: str, context: str) -> Dict[str, Any]:
    """LLM-as-judge faithfulness scoring."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = FAITHFULNESS_PROMPT.format(
        context=context[:3000],
        question=query,
        answer=answer,
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {
            "faithfulness_score": 0.5,
            "reasoning": "Parse error",
            "unsupported_claims": [],
        }


# ── Metric 5: Answer Relevance ───────────────────────────────────────────────
def answer_relevance(query: str, answer: str) -> float:
    """
    Semantic similarity between query and answer.
    High = answer directly addresses the question.
    """
    model = _get_embed_model()
    q_emb = model.encode(query, convert_to_tensor=True)
    a_emb = model.encode(answer, convert_to_tensor=True)
    return round(float(util.cos_sim(q_emb, a_emb)), 4)


# ── Full Single-Pair Eval ─────────────────────────────────────────────────────
def evaluate_single(
    query: str,
    ground_truth: str,
    retrieved_chunks: List[Dict[str, Any]],
    generated_answer: str,
    context: str,
    k: int = 3,
) -> Dict[str, Any]:
    """Run all metrics for a single Q&A pair."""
    pk = precision_at_k(retrieved_chunks, ground_truth, k=k)
    rk = recall_at_k(retrieved_chunks, ground_truth, k=k)
    rr = reciprocal_rank(retrieved_chunks, ground_truth)
    ar = answer_relevance(query, generated_answer)

    time.sleep(3)  # rate limit buffer for Groq
    af = answer_faithfulness(query, generated_answer, context)
    faithfulness = af.get("faithfulness_score", 0.5)

    return {
        "query": query,
        "ground_truth": ground_truth,
        "generated_answer": generated_answer,
        # Retrieval metrics
        "precision_at_k": pk["precision_at_k"],
        "recall_at_k": rk,
        "mrr": rr,
        "max_chunk_score": pk["max_chunk_score"],
        "relevant_chunks": pk["relevant_chunks"],
        # Generation metrics
        "answer_relevance": ar,
        "faithfulness_score": faithfulness,
        "hallucination_rate": round(1.0 - faithfulness, 4),
        "faithfulness_reasoning": af.get("reasoning", ""),
        "query_rewritten": retrieved_chunks[0].get("query_used", query) if retrieved_chunks else query,
    }


# ── Full Suite Runner ─────────────────────────────────────────────────────────
def run_eval_suite(
    eval_pairs: List[Dict[str, Any]],
    chunk_size: int,
    output_prefix: str = "eval/results",
    max_pairs: int = 20,
    use_query_rewriting: bool = False,
) -> pd.DataFrame:
    """
    Run the full eval suite.

    Args:
        eval_pairs: List of {question, answer, title} dicts
        chunk_size: Current chunk size (for labelling output)
        output_prefix: File path prefix for saving results
        max_pairs: Max number of pairs to evaluate
        use_query_rewriting: Whether to rewrite queries before retrieval
    """
    from generation.generate import generate_answer
    from retrieval.retrieve import retrieve, format_context

    os.makedirs("eval", exist_ok=True)
    results = []
    pairs = eval_pairs[:max_pairs]

    label = f"chunk{chunk_size}" + ("_rewrite" if use_query_rewriting else "")
    print(f"\nRunning eval on {len(pairs)} pairs ({label})...")

    for i, pair in enumerate(pairs):
        query = pair["question"]
        gt = pair["answer"]
        print(f"  [{i+1}/{len(pairs)}] {query[:65]}...")

        try:
            chunks = retrieve(query, top_k=3, rewrite=use_query_rewriting)
            context = format_context(chunks)
            gen = generate_answer(query, context)
            answer = gen["answer"]

            row = evaluate_single(query, gt, chunks, answer, context)
            row["chunk_size"] = chunk_size
            row["query_rewriting"] = use_query_rewriting
            results.append(row)

        except Exception as e:
            logger.warning(f"Error on pair {i}: {e}")
            continue

        time.sleep(0.3)

    df = pd.DataFrame(results)

    if not df.empty:
        csv_path = f"{output_prefix}_{label}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    print_summary(df, label)
    return df


def print_summary(df: pd.DataFrame, label: str = ""):
    """Print a clean summary of all eval metrics."""
    sep = "="*55
    print(f"\n{sep}")
    print(f"  EVAL SUMMARY  {label}")
    print(sep)
    if df.empty or "precision_at_k" not in df.columns:
        print("  No results to summarise.")
        print(sep)
        return
    print(f"  Precision@K         : {df['precision_at_k'].mean():.3f}")
    print(f"  Recall@K            : {df['recall_at_k'].mean():.3f}")
    print(f"  MRR                 : {df['mrr'].mean():.3f}")
    print(f"  Answer Relevance    : {df['answer_relevance'].mean():.3f}")
    print(f"  Faithfulness        : {df['faithfulness_score'].mean():.3f}")
    print(f"  Hallucination Rate  : {df['hallucination_rate'].mean():.3f}")
    print(sep)


# ── Before/After Comparison ───────────────────────────────────────────────────
def compare_before_after(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    before_label: str = "Before",
    after_label: str = "After",
) -> pd.DataFrame:
    metrics = [
        "precision_at_k", "recall_at_k", "mrr",
        "answer_relevance", "faithfulness_score", "hallucination_rate"
    ]
    # Only use metrics that exist in both DataFrames
    metrics = [m for m in metrics if m in df_before.columns and m in df_after.columns]

    comparison = pd.DataFrame({
        "Metric": metrics,
        before_label: [df_before[m].mean() for m in metrics],
        after_label: [df_after[m].mean() for m in metrics],
    })
    comparison["Delta"] = comparison[after_label] - comparison[before_label]
    comparison["Improved"] = comparison.apply(
        lambda r: "✅" if (
            (r["Metric"] != "hallucination_rate" and r["Delta"] > 0) or
            (r["Metric"] == "hallucination_rate" and r["Delta"] < 0)
        ) else "—",
        axis=1,
    )
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open("data/eval_pairs.json") as f:
        pairs = json.load(f)
    run_eval_suite(pairs, chunk_size=512, max_pairs=20)

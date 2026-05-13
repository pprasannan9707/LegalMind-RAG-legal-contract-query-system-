"""
eval/evaluator.py
-----------------
Custom evaluation framework using Groq as LLM judge.
Metrics: Precision@K, Recall@K, MRR, Faithfulness, Answer Relevance, Hallucination Rate
"""
import os, json, re, time, logging
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from groq import Groq

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


def _groq_call(prompt: str) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def precision_at_k(retrieved_chunks, ground_truth_answer, k=5, threshold=0.20):
    model = _get_embed_model()
    gt_emb = model.encode(ground_truth_answer, convert_to_tensor=True)
    scores = [float(util.cos_sim(gt_emb, model.encode(c["text"], convert_to_tensor=True))) for c in retrieved_chunks[:k]]
    relevant = sum(1 for s in scores if s >= threshold)
    return {
        "precision_at_k": round(relevant / len(scores), 4) if scores else 0.0,
        "chunk_scores": [round(s, 4) for s in scores],
        "relevant_chunks": relevant,
        "max_chunk_score": round(max(scores), 4) if scores else 0.0,
    }


def recall_at_k(retrieved_chunks, ground_truth_answer, k=5, threshold=0.20):
    model = _get_embed_model()
    gt_emb = model.encode(ground_truth_answer, convert_to_tensor=True)
    for chunk in retrieved_chunks[:k]:
        if float(util.cos_sim(gt_emb, model.encode(chunk["text"], convert_to_tensor=True))) >= threshold:
            return 1.0
    return 0.0


def reciprocal_rank(retrieved_chunks, ground_truth_answer, threshold=0.20):
    model = _get_embed_model()
    gt_emb = model.encode(ground_truth_answer, convert_to_tensor=True)
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if float(util.cos_sim(gt_emb, model.encode(chunk["text"], convert_to_tensor=True))) >= threshold:
            return round(1.0 / rank, 4)
    return 0.0


FAITHFULNESS_PROMPT = """You are evaluating if an AI answer is faithful to the provided context.

Context: {context}
Question: {question}
AI Answer: {answer}

Respond with valid JSON only. No explanation. No markdown. Just this exact structure:
{{"faithfulness_score": 0.8, "reasoning": "one sentence", "unsupported_claims": []}}"""


def answer_faithfulness(query: str, answer: str, context: str) -> Dict[str, Any]:
    default = {"faithfulness_score": 0.5, "reasoning": "parse error", "unsupported_claims": []}
    try:
        raw = _groq_call(FAITHFULNESS_PROMPT.format(
            context=context[:2000], question=query, answer=answer
        ))

        # Clean markdown fences
        clean = re.sub(r"```json|```", "", raw).strip()

        # Try 1: full JSON parse
        try:
            match = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        # Try 2: greedy JSON parse
        try:
            return json.loads(clean)
        except Exception:
            pass

        # Try 3: regex extract just the number
        score_match = re.search(r'faithfulness_score[^0-9]*([0-9]+\.?[0-9]*)', clean)
        if score_match:
            return {
                "faithfulness_score": float(score_match.group(1)),
                "reasoning": "score extracted from partial response",
                "unsupported_claims": [],
            }

        return default

    except Exception as e:
        logger.warning(f"Faithfulness judge failed: {e}")
        return default


def answer_relevance(query: str, answer: str) -> float:
    model = _get_embed_model()
    return round(float(util.cos_sim(
        model.encode(query, convert_to_tensor=True),
        model.encode(answer, convert_to_tensor=True)
    )), 4)


def evaluate_single(query, ground_truth, retrieved_chunks, generated_answer, context, k=5):
    pk = precision_at_k(retrieved_chunks, ground_truth, k=k)
    rk = recall_at_k(retrieved_chunks, ground_truth, k=k)
    rr = reciprocal_rank(retrieved_chunks, ground_truth)
    ar = answer_relevance(query, generated_answer)
    af = answer_faithfulness(query, generated_answer, context)
    faith = af.get("faithfulness_score", 0.5)
    return {
        "query": query, "ground_truth": ground_truth, "generated_answer": generated_answer,
        "precision_at_k": pk["precision_at_k"], "recall_at_k": rk, "mrr": rr,
        "max_chunk_score": pk["max_chunk_score"], "relevant_chunks": pk["relevant_chunks"],
        "answer_relevance": ar, "faithfulness_score": faith,
        "hallucination_rate": round(1.0 - faith, 4),
        "faithfulness_reasoning": af.get("reasoning", ""),
        "query_rewritten": retrieved_chunks[0].get("query_used", query) if retrieved_chunks else query,
    }


def run_eval_suite(eval_pairs, chunk_size, output_prefix="eval/results", max_pairs=10, use_query_rewriting=False):
    from generation.generate import generate_answer
    from retrieval.retrieve import retrieve, format_context
    os.makedirs("eval", exist_ok=True)
    results = []
    label = f"chunk{chunk_size}" + ("_rewrite" if use_query_rewriting else "")
    print(f"\nRunning eval on {min(len(eval_pairs), max_pairs)} pairs ({label})...")
    for i, pair in enumerate(eval_pairs[:max_pairs]):
        query, gt = pair["question"], pair["answer"]
        print(f"  [{i+1}/{max_pairs}] {query[:65]}...")
        try:
            chunks = retrieve(query, top_k=5, rewrite=use_query_rewriting)
            context = format_context(chunks)
            answer = generate_answer(query, context)["answer"]
            row = evaluate_single(query, gt, chunks, answer, context)
            row.update({"chunk_size": chunk_size, "query_rewriting": use_query_rewriting})
            results.append(row)
        except Exception as e:
            logger.warning(f"Error on pair {i}: {e}")
        time.sleep(2)
    df = pd.DataFrame(results)
    if not df.empty:
        path = f"{output_prefix}_{label}.csv"
        df.to_csv(path, index=False)
        print(f"Results saved to {path}")
    print_summary(df, label)
    return df


def print_summary(df, label=""):
    sep = "="*55
    print(f"\n{sep}\n  EVAL SUMMARY  {label}\n{sep}")
    if df.empty or "precision_at_k" not in df.columns:
        print("  No results to summarise.")
        print(sep)
        return
    for metric, name in [
        ("precision_at_k", "Precision@K"), ("recall_at_k", "Recall@K"),
        ("mrr", "MRR"), ("answer_relevance", "Answer Relevance"),
        ("faithfulness_score", "Faithfulness"), ("hallucination_rate", "Hallucination Rate")
    ]:
        print(f"  {name:<22}: {df[metric].mean():.3f}")
    print(sep)


def compare_before_after(df_before, df_after, before_label="Before", after_label="After"):
    metrics = ["precision_at_k", "recall_at_k", "mrr", "answer_relevance", "faithfulness_score", "hallucination_rate"]
    metrics = [m for m in metrics if m in df_before.columns and m in df_after.columns]
    comp = pd.DataFrame({
        "Metric": metrics,
        before_label: [df_before[m].mean() for m in metrics],
        after_label: [df_after[m].mean() for m in metrics],
    })
    comp["Delta"] = comp[after_label] - comp[before_label]
    comp["Improved"] = comp.apply(
        lambda r: "✅" if (r["Metric"] != "hallucination_rate" and r["Delta"] > 0)
                      or (r["Metric"] == "hallucination_rate" and r["Delta"] < 0)
                  else "—", axis=1)
    return comp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open("data/eval_pairs.json") as f:
        pairs = json.load(f)
    run_eval_suite(pairs, chunk_size=512, max_pairs=20)

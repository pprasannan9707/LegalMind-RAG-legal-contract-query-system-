"""
generation/generate.py
-----------------------
Uses Groq LLM (LLaMA 3) to generate grounded answers from retrieved legal context.
Includes citation tracking and hallucination-aware prompting.
"""

import os
from typing import List, Dict, Any, Tuple
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"   # fast + free on Groq
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are LegalMind, an expert AI legal document assistant.

Your job is to answer questions about legal contracts and documents using ONLY the provided context.

Rules you must follow:
1. Base your answer strictly on the context provided. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say: "The provided documents do not contain sufficient information to answer this question."
3. Always cite which source (Source 1, Source 2, etc.) supports your answer.
4. Be precise and use legal terminology correctly.
5. Do not speculate or make assumptions beyond what is written.
"""


def build_prompt(query: str, context: str) -> str:
    return f"""CONTEXT FROM LEGAL DOCUMENTS:
{context}

QUESTION:
{query}

Please answer the question based only on the context above. Cite the relevant source(s)."""


def generate_answer(
    query: str,
    context: str,
    model: str = GROQ_MODEL,
) -> Dict[str, Any]:
    """
    Call Groq LLM to generate an answer grounded in the provided context.

    Returns:
        dict with keys: answer, model, prompt_tokens, completion_tokens
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = build_prompt(query, context)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0,   # deterministic for eval reproducibility
    )

    answer = response.choices[0].message.content.strip()
    usage = response.usage

    return {
        "answer": answer,
        "model": model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


def rag_pipeline(
    query: str,
    top_k: int = 3,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full RAG pipeline: retrieve → format context → generate answer.

    Returns:
        (answer_text, retrieved_chunks)
    """
    from retrieval.retrieve import retrieve, format_context

    chunks = retrieve(query, top_k=top_k)
    context = format_context(chunks)
    result = generate_answer(query, context)
    return result["answer"], chunks


if __name__ == "__main__":
    query = "What are the termination clauses in this agreement?"
    answer, chunks = rag_pipeline(query)
    print(f"\n🔍 Query: {query}")
    print(f"\n💡 Answer:\n{answer}")
    print(f"\n📚 Retrieved {len(chunks)} chunks")

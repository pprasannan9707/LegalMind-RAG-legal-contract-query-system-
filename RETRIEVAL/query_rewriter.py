"""
retrieval/query_rewriter.py
----------------------------
Improvement 1: Query Rewriting

WHY: Legal documents use formal, precise terminology. Users ask questions
in plain English ("can they fire me?"). The embedding model struggles to
bridge this vocabulary gap. A lightweight rewrite step converts the query
into legal terminology BEFORE embedding, so retrieval finds the right clauses.

DESIGN DECISION: One Groq call, deterministic (temperature=0), cached by
query string. Simple dict cache — no Redis, no overhead. If rewriting fails,
we fall back to the original query silently. Zero demo risk.

INTENTIONALLY NOT ADDED:
- HyDE (generates a full hypothetical document — 3x more tokens, harder to debug)
- Multi-query retrieval (multiple DB calls — slower, more complex merging logic)
- Query expansion with synonyms (good recall boost but adds noise to precision)
These are documented as "next steps" in the written doc.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

# Simple in-memory cache — avoids re-calling Groq for repeated queries
_rewrite_cache: dict = {}

REWRITE_PROMPT = """You are a legal document retrieval expert.

Rewrite the following user query into precise legal terminology that would appear in a commercial contract or legal agreement. 

Rules:
- Use formal legal language (e.g. "termination" not "firing", "indemnification" not "protection", "governing law" not "which law applies")
- Keep it concise — one sentence maximum
- Output ONLY the rewritten query, nothing else

User query: {query}

Rewritten legal query:"""


def rewrite_query(query: str, use_cache: bool = True) -> str:
    """
    Rewrite a plain-English legal query into formal legal terminology.
    Falls back to original query on any error.

    Args:
        query: User's natural language question
        use_cache: Whether to cache rewrites (default True)

    Returns:
        Rewritten query string (or original if rewriting fails)
    """
    if use_cache and query in _rewrite_cache:
        logger.debug(f"Cache hit for query rewrite: {query[:50]}")
        return _rewrite_cache[query]

    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": REWRITE_PROMPT.format(query=query)}
            ],
            max_tokens=100,
            temperature=0.0,  # deterministic
        )
        rewritten = response.choices[0].message.content.strip()

        # Sanity check — if LLM returns something too long or weird, use original
        if len(rewritten) > 300 or len(rewritten) < 5:
            logger.warning(f"Query rewrite sanity check failed, using original. Got: {rewritten[:100]}")
            return query

        if use_cache:
            _rewrite_cache[query] = rewritten

        logger.info(f"Query rewritten: '{query[:60]}' -> '{rewritten[:60]}'")
        return rewritten

    except Exception as e:
        # Silent fallback — never break the pipeline over a rewrite failure
        logger.warning(f"Query rewriting failed: {e}. Using original query.")
        return query


def rewrite_batch(queries: list, use_cache: bool = True) -> list:
    """Rewrite a list of queries. Returns list of (original, rewritten) tuples."""
    return [(q, rewrite_query(q, use_cache=use_cache)) for q in queries]


if __name__ == "__main__":
    # Quick test
    test_queries = [
        "What happens if they fire me?",
        "Can I take my work to another company?",
        "Who owns the stuff I make at work?",
        "What if there's a disagreement?",
        "How long does this contract last?",
    ]
    print("Query Rewriting Demo\n" + "="*50)
    for q in test_queries:
        rewritten = rewrite_query(q)
        print(f"Original : {q}")
        print(f"Rewritten: {rewritten}")
        print()

"""
run.py
------
Master CLI for LegalMind RAG pipeline.

Usage:
  python run.py --step ingest          # Load CUAD, chunk, embed, store
  python run.py --step eval            # Baseline eval (no rewriting)
  python run.py --step eval-rewrite    # Eval WITH query rewriting
  python run.py --step fix             # Failure mode 1 fix (chunk size)
  python run.py --step demo-rewrite    # Demo query rewriting in terminal
  python run.py --step app             # Launch Streamlit UI
"""

import argparse
import subprocess
import sys
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

os.makedirs("data", exist_ok=True)
os.makedirs("eval", exist_ok=True)


def _load_pairs():
    path = "data/eval_pairs.json"
    if not os.path.exists(path):
        print("eval_pairs.json not found — run --step ingest first.")
        sys.exit(1)
    with open(path) as f:
        pairs = json.load(f)
    if not pairs:
        print("No eval pairs found. Check ingestion ran correctly.")
        sys.exit(1)
    print(f"Loaded {len(pairs)} eval pairs.")
    return pairs


def step_ingest():
    print("\n" + "="*55)
    print("STEP 1: Ingestion")
    print("NOTE: First run downloads BAAI/bge-large-en-v1.5 (~1.3GB)")
    print("="*55)
    from ingestion.ingest import load_cuad, save_eval_data, build_chroma_collection, CHUNK_SIZE
    docs, eval_pairs = load_cuad()
    save_eval_data(eval_pairs)
    build_chroma_collection(docs, chunk_size=CHUNK_SIZE)
    print("Ingestion complete.")


def step_eval(use_rewriting=False):
    label = "WITH query rewriting" if use_rewriting else "baseline (no rewriting)"
    print("\n" + "="*55)
    print(f"STEP 2: Evaluation — {label}")
    print("="*55)
    pairs = _load_pairs()
    from eval.evaluator import run_eval_suite
    run_eval_suite(pairs, chunk_size=512, max_pairs=5, use_query_rewriting=use_rewriting)
    print("Eval complete.")


def step_fix():
    print("\n" + "="*55)
    print("STEP 3: Failure Modes + Fix")
    print("="*55)
    pairs = _load_pairs()
    from eval.failure_modes import demonstrate_semantic_mismatch, run_failure_mode_1_fix
    demonstrate_semantic_mismatch()
    run_failure_mode_1_fix(pairs, max_pairs=5)
    print("Failure mode analysis complete.")


def step_demo_rewrite():
    print("\n" + "="*55)
    print("STEP: Query Rewriting Demo")
    print("="*55)
    from retrieval.query_rewriter import rewrite_query
    queries = [
        "What happens if they fire me without notice?",
        "Can I share this deal with a competitor?",
        "Who owns the software I build for them?",
        "What if we have a disagreement?",
        "How long are we locked in for?",
        "What are the payment terms?",
    ]
    print(f"\n{'Original Query':<48} {'Rewritten (Legal)'}")
    print("-" * 100)
    for q in queries:
        rewritten = rewrite_query(q)
        print(f"{q:<48} {rewritten}")


def step_app():
    print("\n" + "="*55)
    print("STEP: Launching Streamlit App")
    print("="*55)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "interface/app.py"])


def main():
    parser = argparse.ArgumentParser(description="LegalMind RAG Pipeline")
    parser.add_argument(
        "--step",
        choices=["ingest", "eval", "eval-rewrite", "fix", "demo-rewrite", "app", "all"],
        default="app",
        help="Which step to run",
    )
    args = parser.parse_args()

    steps = {
        "ingest": lambda: step_ingest(),
        "eval": lambda: step_eval(use_rewriting=False),
        "eval-rewrite": lambda: step_eval(use_rewriting=True),
        "fix": lambda: step_fix(),
        "demo-rewrite": lambda: step_demo_rewrite(),
        "app": lambda: step_app(),
        "all": lambda: [step_ingest(), step_eval(False), step_eval(True), step_fix(), step_app()],
    }

    steps[args.step]()


if __name__ == "__main__":
    main()

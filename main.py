#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for the Deep Research Agent.

Usage
-----
Basic query:
    python main.py "What are the key differences between RLHF and DPO?"

Pre-load a text file into the knowledge base first:
    python main.py --ingest paper.txt "Summarise the paper's main contributions"

Custom budget / model:
    python main.py --cap 30000 --model claude-sonnet-4-6 "Your question"

The agent writes every run to evaluation.md automatically.
"""

import argparse
import sys
import os

from dotenv import load_dotenv
load_dotenv()

from agent import DeepResearchAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep Research Agent — Vector RAG + Episodic Buffer + Summary Cascade"
    )
    p.add_argument("query", help="Research question to answer")
    p.add_argument(
        "--ingest", metavar="FILE",
        help="Plain-text file to load into the vector knowledge base before querying",
    )
    p.add_argument(
        "--cap", type=int, default=50_000,
        help="Token budget cap per query (default: 50,000)",
    )
    p.add_argument(
        "--model", default="claude-opus-4-6",
        help="Claude model ID (default: claude-opus-4-6)",
    )
    p.add_argument(
        "--chroma", default="./chroma_db",
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    p.add_argument(
        "--eval", default="evaluation.md",
        help="Evaluation log path (default: evaluation.md)",
    )
    p.add_argument(
        "--max-iter", type=int, default=8,
        help="Maximum agent iterations (default: 8)",
    )
    return p.parse_args()


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks for ingestion."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def main() -> None:
    args = parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it or add it to a .env file in this directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    agent = DeepResearchAgent(
        token_cap=args.cap,
        model=args.model,
        chroma_path=args.chroma,
        eval_path=args.eval,
        max_iterations=args.max_iter,
    )

    # ── Optional: pre-load a knowledge file ───────────────────────────
    if args.ingest:
        print(f"Ingesting {args.ingest} …", flush=True)
        with open(args.ingest, "r", encoding="utf-8") as fh:
            raw = fh.read()
        chunks = chunk_text(raw)
        for i, chunk in enumerate(chunks, 1):
            agent.ingest(chunk, metadata={"file": args.ingest, "chunk": i})
        print(f"  Ingested {len(chunks)} chunks.\n", flush=True)

    # ── Run the query ──────────────────────────────────────────────────
    print(f"Query: {args.query}\n", flush=True)
    print("=" * 60, flush=True)

    answer = agent.query(args.query)

    print("\n" + "=" * 60)
    print("ANSWER\n")
    print(answer)
    print()
    print(f"(Run logged to {args.eval})")


if __name__ == "__main__":
    main()

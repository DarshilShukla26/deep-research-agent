#!/usr/bin/env python3
"""FastAPI HTTP server — exposes the Deep Research Agent over REST.

Start with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET  /health       — liveness probe
POST /query        — run a research query (returns answer + metadata)
POST /ingest       — add text chunks to the vector knowledge base
GET  /stats        — memory layer statistics
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from agent import DeepResearchAgent


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deep Research Agent",
    description=(
        "3-layer memory research agent: "
        "Vector RAG (ChromaDB) + Episodic Buffer + Summary Cascade"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., description="Research question to answer")
    cap: int = Field(50_000, ge=1_000, le=200_000, description="Token budget cap")
    model: str = Field("claude-opus-4-6", description="Claude model ID")
    max_iterations: int = Field(8, ge=1, le=20, description="Max agent loop iterations")


class QueryResponse(BaseModel):
    answer: str
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost_usd: float
    memory_strategies: list[str]
    sub_questions: list[str]
    iterations: int
    budget_utilisation_pct: float
    self_score: dict = {}


class IngestRequest(BaseModel):
    text: str = Field(..., description="Text to store in the vector knowledge base")
    source: Optional[str] = Field(None, description="Source label")
    chunk_size: int = Field(800, ge=50, description="Words per chunk")
    overlap: int = Field(100, ge=0, description="Word overlap between consecutive chunks")


class IngestResponse(BaseModel):
    chunks_ingested: int
    doc_ids: list[str]


class StatsResponse(BaseModel):
    vector_store_count: int
    episodic_buffer_turns: int
    has_summary: bool
    summary_preview: str


# ── Agent cache ───────────────────────────────────────────────────────────────
_agents: dict[str, DeepResearchAgent] = {}


def _get_agent(cap: int, model: str, max_iterations: int) -> DeepResearchAgent:
    key = f"{cap}:{model}:{max_iterations}"
    if key not in _agents:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY is not set on the server.",
            )
        _agents[key] = DeepResearchAgent(
            token_cap=cap,
            model=model,
            max_iterations=max_iterations,
        )
    return _agents[key]


def _chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — returns 200 if the server is running."""
    return {"status": "ok", "service": "deep-research-agent"}


@app.post("/query", response_model=QueryResponse, tags=["agent"])
def query(req: QueryRequest):
    """
    Run a research query through the full 3-layer memory pipeline.

    The agent will:
    1. Fetch relevant chunks from the vector store (RAG)
    2. Inject recent turns from the episodic buffer
    3. Include the rolling summary if the buffer has previously overflowed
    4. Call Claude with tool use (decompose → search → synthesize)
    5. Log the run to evaluation.md
    """
    agent = _get_agent(req.cap, req.model, req.max_iterations)
    result = agent.query_full(req.query)
    return QueryResponse(**result)


@app.post("/query/pretty", tags=["agent"])
def query_pretty(req: QueryRequest):
    """
    Same as /query but returns a human-readable plain-text response
    instead of JSON — easy to read directly in the terminal.
    """
    from fastapi.responses import PlainTextResponse
    agent = _get_agent(req.cap, req.model, req.max_iterations)
    result = agent.query_full(req.query)

    divider = "─" * 60
    strategies = ", ".join(result["memory_strategies"]) or "none"
    sub_qs = result["sub_questions"]

    lines = [
        divider,
        f"  QUERY   : {req.query}",
        divider,
        "",
        result["answer"],
        "",
        divider,
        "  RUN STATS",
        divider,
        f"  Model      : {req.model}",
        f"  Tokens     : {result['tokens_total']:,} / {req.cap:,}  ({result['budget_utilisation_pct']}% used)",
        f"  Cost       : ${result['cost_usd']:.6f}",
        f"  Memory     : {strategies}",
        f"  Iterations : {result['iterations']}",
    ]
    if sub_qs:
        lines.append(f"  Sub-questions decomposed ({len(sub_qs)}):")
        for i, q in enumerate(sub_qs, 1):
            lines.append(f"    {i}. {q}")
    lines.append(divider)

    return PlainTextResponse("\n".join(lines))


@app.post("/ingest", response_model=IngestResponse, tags=["knowledge"])
def ingest(req: IngestRequest):
    """
    Chunk and store text in the persistent ChromaDB vector store.
    Duplicate chunks (by SHA-256 hash) are silently ignored.
    """
    agent = _get_agent(50_000, "claude-opus-4-6", 8)
    chunks = _chunk_text(req.text, req.chunk_size, req.overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No non-empty chunks produced from input text.")
    ids = []
    for i, chunk in enumerate(chunks):
        meta = {"source": req.source or "api", "chunk": i + 1}
        doc_id = agent.ingest(chunk, metadata=meta)
        ids.append(doc_id)
    return IngestResponse(chunks_ingested=len(chunks), doc_ids=ids)


@app.get("/stats", response_model=StatsResponse, tags=["ops"])
def stats():
    """Return current memory layer statistics (no LLM call made)."""
    if not _agents:
        return StatsResponse(
            vector_store_count=0,
            episodic_buffer_turns=0,
            has_summary=False,
            summary_preview="",
        )
    agent = next(iter(_agents.values()))
    summary = agent._cascade.get_summary()
    return StatsResponse(
        vector_store_count=agent._rag.count(),
        episodic_buffer_turns=len(agent._buf),
        has_summary=bool(summary),
        summary_preview=summary[:200] if summary else "",
    )

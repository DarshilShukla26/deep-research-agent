# Deep Research Agent — Evaluation Log

---

## Architecture & Design Trade-offs

### Overview

The agent is built around a **3-layer memory system** that mirrors the
`g3_deep_research_agent_architecture.svg` reference diagram.  Each layer
targets a different time-horizon of knowledge, and the **Context Assembler**
combines all three within a hard token budget before every Claude call.

```
User query
    │
Router + Budget Guard        ← enforces token_cap per query (default 50 000)
    │
Memory Layer (parallel fetch)
  ├── Vector RAG             ← long-term semantic store  (ChromaDB)
  ├── Episodic Buffer        ← short-term ring of recent turns
  └── Summary Cascade        ← compressed history when buffer overflows
    │
Context Assembler            ← greedy bin-pack to stay under budget
    │
Claude (tool use)            ← decompose → search → synthesize loop
    │
evaluation.md logger         ← tokens, cost, strategies, sub-questions
    │
Response to user
```

---

### Layer 1 — Vector RAG (`memory/vector_rag.py`)

**What it is:** A persistent ChromaDB collection with `all-MiniLM-L6-v2`
embeddings.  Every ingested document chunk and every agent-discovered fact is
stored here with a SHA-256 content hash as its ID (so duplicates are silently
ignored).

**Why ChromaDB over a hosted vector DB (Pinecone, Weaviate, etc.):**

| Criterion | ChromaDB (local) | Hosted vector DB |
|---|---|---|
| Latency | < 5 ms (in-process) | 50–200 ms network round-trip |
| Cost | Free | $0.10–$1.00 / 1M vectors / month |
| Privacy | Data never leaves disk | Data sent to third-party |
| Scalability | Single machine | Horizontal, cloud-native |
| Setup complexity | One `pip install` | API keys, account, network config |

**Chosen because** the agent is designed to run locally with zero external
dependencies beyond the Anthropic API.  For a production multi-user system,
swapping `VectorRAG` for a hosted store is a one-file change.

**Trade-off accepted:** ChromaDB's default embedding model (`all-MiniLM-L6-v2`)
is weaker than OpenAI `text-embedding-3-large` or Cohere `embed-v3`.  Retrieval
recall on highly technical queries may be lower.  This is offset by the
agent's `search_knowledge` tool, which lets Claude issue targeted sub-queries
rather than relying on a single broad semantic search.

---

### Layer 2 — Episodic Buffer (`memory/episodic_buffer.py`)

**What it is:** A fixed-capacity ring buffer (default **10 turns**) of
`(query, answer)` pairs from the current session, plus a running log of all
sub-questions the agent has decomposed.

**Why a ring buffer instead of full conversation history:**

| Approach | Tokens consumed | Risk |
|---|---|---|
| Full history appended | O(n × avg_turn_length) — grows unbounded | Blows past token cap in long sessions |
| Ring buffer (last N turns) | O(N × avg_turn_length) — constant | Oldest turns silently lost |
| Ring buffer + cascade | O(N) + O(summary) — bounded | Small compression cost |

A raw conversation passed to Claude as `messages[]` costs tokens on every
call.  The ring buffer externalises that cost: only the last N turns are
ever injected into the context, keeping per-call overhead predictable.

**Trade-off accepted:** Turns evicted from the ring are permanently lost
*unless* the Summary Cascade has already compressed them (see Layer 3).  If
the cascade is skipped due to budget pressure, fine-grained detail from early
turns may be unrecoverable.

---

### Layer 3 — Summary Cascade (`memory/summary_cascade.py`)

**What it is:** When the episodic buffer reaches capacity, the **5 oldest
turns** are drained and folded into a rolling ≤200-word summary via a
dedicated Claude call.  The summary is prepended to every subsequent context
window.

**Why a cascade (incremental summarisation) over a fixed sliding window:**

| Strategy | Context size | Detail preserved | LLM calls for summary |
|---|---|---|---|
| Sliding window (last N tokens) | Constant | Recent only | 0 (no summary) |
| Full summarisation on demand | Spiky | High initially, then compressed | 1 per flush |
| **Cascade (rolling summary)** | Constant | Accumulated across all flushes | 1 per flush |
| Hierarchical summary tree | Constant | High | Multiple |

A rolling summary lets the agent carry a *distilled narrative* across
arbitrarily long sessions without proportional token growth.  The cascade
calls a cheaper model variant if cost is a concern (configurable via
`--model`).

**Budget-aware fallback:** If the cascade call would itself exceed the
remaining token budget, the system falls back to a simple string concatenation
truncated to 1,200 characters.  This guarantees the agent never overspends on
memory management.

**Trade-off accepted:** Summarisation is lossy.  Specific numbers, verbatim
quotes, and rare details mentioned early in a session may be paraphrased away.
For research tasks requiring exact citation this is a limitation; for
synthesising thematic understanding it is acceptable.

---

### Token Budget Guard (`budget_guard.py`)

**Self-defined constraint:** The agent enforces a hard **token cap per query**
(default `50,000`; override with `--cap`).  This cap covers *all* tokens
spent during a query — decomposition calls, search calls, synthesis calls, and
the summary cascade — not just the final answer call.

**Enforcement mechanism:**

1. `BudgetGuard.new_budget()` creates a fresh `TokenBudget` dataclass at the
   start of each `agent.query()` call.
2. Every `client.messages.create()` response records its actual
   `usage.input_tokens` and `usage.output_tokens` back into the budget.
3. Before each Claude call: `if budget.remaining < 1_500: break` — the loop
   exits gracefully rather than hard-erroring.
4. `ContextAssembler` reserves `RESPONSE_RESERVE = 4,096` tokens before
   bin-packing context, so the response generation is never starved.

**Cost estimation:** `TokenBudget.cost_usd(model)` applies published per-1M
token rates so every run log includes a dollar figure.

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| claude-opus-4-6 | $5.00 | $25.00 |
| claude-sonnet-4-6 | $3.00 | $15.00 |
| claude-haiku-4-5 | $1.00 | $5.00 |

**Trade-off accepted:** `tiktoken` (`cl100k_base`) is used for pre-call
estimation; Claude uses its own internal tokeniser.  Counts may differ by
±2–5%.  The 1,500-token safety margin absorbs this discrepancy in practice.

---

### Context Assembler (`context_assembler.py`)

**What it is:** A greedy bin-packing algorithm that builds the context string
from the three memory layers within the current token allowance
(`budget.remaining − RESPONSE_RESERVE`).

**Priority order (highest → lowest):**

1. **Historical Summary** — most compressed, lowest token cost, highest
   signal density.  Always included first if it exists.
2. **Episodic Buffer** — recent session turns.  Included second because
   recency matters more than semantic similarity for follow-up questions.
3. **Vector RAG chunks** — iterated greedily; shorter/closer chunks included
   first until the allowance is exhausted.  A later shorter chunk can still
   be included even if an earlier one was skipped.

**Why greedy over optimal (knapsack)?**  For ≤50 RAG chunks the greedy
approximation is within a few percent of optimal and runs in O(n) vs O(n·W)
for the DP solution.  Given that each chunk selection runs inside the hot
path of the agent loop, latency matters more than packing perfection.

**Trade-off accepted:** If the Summary + Episodic sections are large,
RAG chunks may be partially or fully squeezed out.  This is by design: a
concise summary of 10 turns is more useful than 3 semantically distant
chunks.

---

### Sub-Question Decomposition

The `decompose_question` tool lets Claude break a complex query into an
ordered list of focused sub-questions.  Each sub-question is:

- Registered in the episodic buffer's sub-question log (for traceability)
- Eagerly searched in the vector store (pre-warm cache)
- Looped back through the full memory → context → Claude pipeline

**Why tool-based decomposition over prompt-based chain-of-thought?**

| Approach | Structured output | Loopable | Token cost |
|---|---|---|---|
| Chain-of-thought (CoT) | No (free text) | No | Moderate |
| Tool call (`decompose_question`) | Yes (JSON array) | Yes | Same |
| Dedicated planner LM call | Yes | Yes | Extra call |

Tool-based decomposition is free — it happens within the existing Claude
call — and produces a machine-readable list that the agent loop can iterate
over deterministically.

---

### What Was Not Built (and Why)

| Spec item | Decision |
|---|---|
| **n8n / Dify workflow layer** | Replaced by native Python orchestration in `agent.py`.  n8n/Dify add a visual UI and webhook triggers useful for no-code teams; for a code-first project they add infrastructure overhead with no functional gain.  The Python loop is easier to test, version, and extend. |
| **Web search tool** | Out of scope for this submission; the `search_knowledge` tool is the retrieval primitive.  A `web_search` tool could be added as a fifth entry in `TOOLS` without changing any other component. |
| **Streaming responses** | Not implemented; `client.messages.create()` is used in blocking mode.  Adding `stream=True` would only affect the CLI output experience, not the memory or budget logic. |

---

Each section below this line is one **run log** appended automatically by
`evaluator.py` when `python main.py` is executed.

---

## Run — 2026-04-05 17:05:31

**Query:** What are the key differences between RLHF and DPO?

### Token Usage
| Field | Value |
|---|---|
| Model | `claude-opus-4-6` |
| Input tokens | 6,701 |
| Output tokens | 3,012 |
| Total tokens | 9,713 |
| Token cap | 50,000 |
| Utilisation | 19.4% |

### Cost Estimate
| Component | Tokens | Rate ($/1M) | Cost ($) |
|---|---|---|---|
| Input | 6,701 | 5.00 | 0.033505 |
| Output | 3,012 | 25.00 | 0.075300 |
| **Total** | | | **0.108805** |

### Memory Strategies Triggered
- vector_rag

### Sub-Questions Decomposed
- What is RLHF (Reinforcement Learning from Human Feedback) and how does it work?
- What is DPO (Direct Preference Optimization) and how does it work?
- What are the architectural and pipeline differences between RLHF and DPO?
- What are the training stability and computational cost differences between RLHF and DPO?
- What are the performance and practical trade-off differences between RLHF and DPO?

**Iterations:** 3

---

## Run — 2026-04-05 18:00:27

**Query:** What is the transformer architecture?

### Token Usage
| Field | Value |
|---|---|
| Model | `claude-opus-4-6` |
| Input tokens | 11,236 |
| Output tokens | 2,134 |
| Total tokens | 13,370 |
| Token cap | 15,000 |
| Utilisation | 89.1% |

### Cost Estimate
| Component | Tokens | Rate ($/1M) | Cost ($) |
|---|---|---|---|
| Input | 11,236 | 5.00 | 0.056180 |
| Output | 2,134 | 25.00 | 0.053350 |
| **Total** | | | **0.109530** |

### Memory Strategies Triggered
- vector_rag

**Iterations:** 3

---

## Run — 2026-04-05 18:01:58

**Query:** Describe the evolution of AI, what was the starting point?

### Token Usage
| Field | Value |
|---|---|
| Model | `claude-opus-4-6` |
| Input tokens | 14,858 |
| Output tokens | 1,496 |
| Total tokens | 16,354 |
| Token cap | 15,000 |
| Utilisation | 109.0% |

### Cost Estimate
| Component | Tokens | Rate ($/1M) | Cost ($) |
|---|---|---|---|
| Input | 14,858 | 5.00 | 0.074290 |
| Output | 1,496 | 25.00 | 0.037400 |
| **Total** | | | **0.111690** |

### Memory Strategies Triggered
- episodic_buffer
- vector_rag

### Sub-Questions Decomposed
- What was the starting point / origin of AI as a field?
- What were the key early milestones in AI (1950s-1970s)?
- How did AI evolve through the 'AI winters' and expert systems era (1970s-1990s)?
- How did machine learning and deep learning drive modern AI (1990s-2010s)?
- What are the major breakthroughs in the current era of AI (2017-present, transformers, LLMs, generative AI)?

**Iterations:** 3

---

## Run — 2026-04-05 18:04:12

**Query:** Describe the evolution of AI, what was the starting point?

### Token Usage
| Field | Value |
|---|---|
| Model | `claude-opus-4-6` |
| Input tokens | 11,753 |
| Output tokens | 1,867 |
| Total tokens | 13,620 |
| Token cap | 50,000 |
| Utilisation | 27.2% |

### Cost Estimate
| Component | Tokens | Rate ($/1M) | Cost ($) |
|---|---|---|---|
| Input | 11,753 | 5.00 | 0.058765 |
| Output | 1,867 | 25.00 | 0.046675 |
| **Total** | | | **0.105440** |

### Memory Strategies Triggered
- vector_rag

**Iterations:** 2

---


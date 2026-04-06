# Deep Research Agent

A research agent that actually remembers what it has learned. You ask it a complex question, it breaks it down into sub-questions, searches its memory, looks things up on the web if needed, and gives you a well-structured answer — all while staying within a token budget you define.

Built with Python, ChromaDB, and the Anthropic SDK. Comes with a REST API and an n8n workflow for routing and automation.

---

## What it does

- Breaks multi-part questions into focused sub-questions automatically
- Searches a local vector knowledge base (ChromaDB) for relevant context
- Falls back to live web search (DuckDuckGo) when the knowledge base has nothing useful
- Enforces a per-query token budget so you never get surprise costs
- Scores its own answers after every run (completeness, clarity, accuracy)
- Logs every run to `evaluation.md` with token counts, cost, and memory strategies used

---

## Architecture

```
User query
    │
Router + Budget Guard      ← hard token cap per query (default 50,000)
    │
Memory Layer
  ├── Vector RAG            ← ChromaDB semantic search
  ├── Episodic Buffer       ← last 10 turns of this session
  └── Summary Cascade       ← compressed history (kicks in when buffer fills up)
    │
Context Assembler           ← fits memory into the remaining token budget
    │
Claude (tool use)           ← haiku for search/decompose, opus for final answer
  ├── decompose_question
  ├── search_knowledge
  ├── search_web
  ├── add_to_knowledge
  └── synthesize_answer
    │
evaluation.md logger        ← tokens, cost, memory strategies, self-score
```

---

## Quick Start

**1. Clone and enter the project**
```bash
git clone https://github.com/DarshilShukla26/deep-research-agent.git
cd deep-research-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your API key**
```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

**4. Run a query**
```bash
python3 main.py "What are the key differences between RLHF and DPO?"
```

That's it. The answer prints to your terminal and the run is logged to `evaluation.md`.

---

## CLI Usage

```bash
# Basic query
python3 main.py "your question"

# Pre-load a text file into the knowledge base, then query it
python3 main.py --ingest myfile.txt "Summarise the main contributions"

# Set a custom token budget
python3 main.py --cap 20000 "Quick summary of attention mechanisms"

# Use a cheaper model
python3 main.py --model claude-haiku-4-5 "What is RLHF?"

# Compare the same question across 3 budget tiers
python3 main.py --compare "Compare RLHF and DPO"

# Compare with custom budgets
python3 main.py --compare --caps 10000,25000,50000 "Explain transformers"
```

### All CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--cap` | 50000 | Token budget cap for this query |
| `--model` | claude-opus-4-6 | Claude model for final synthesis |
| `--ingest` | — | Text file to load into the knowledge base first |
| `--compare` | off | Run query at multiple budgets and compare |
| `--caps` | 15000,30000,50000 | Budget tiers for --compare mode |
| `--chroma` | ./chroma_db | ChromaDB persistence directory |
| `--eval` | evaluation.md | Evaluation log file path |
| `--max-iter` | 8 | Max agent loop iterations |

---

## REST API

Start the server:
```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs at **http://localhost:8000/docs**

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Liveness check |
| `/query` | POST | Run a query, returns JSON with answer + metadata |
| `/query/pretty` | POST | Same but returns readable plain text |
| `/ingest` | POST | Add text to the knowledge base |
| `/stats` | GET | Memory layer statistics |

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the transformer architecture?", "cap": 30000}'
```

**Pretty output:**
```bash
curl -X POST http://localhost:8000/query/pretty \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain attention mechanisms", "cap": 30000}'
```

---

## n8n Workflow

The included `n8n_workflow.json` adds smart query routing on top of the API.

**What it adds:**
- Auto-classifies every query as `simple` (15k), `complex` (30k), or `deep` (50k) tokens
- Posts results to Slack (optional)
- Handles errors gracefully with a separate error branch

**Setup:**
1. Start the API server (see above)
2. Install and start n8n:
   ```bash
   npm install -g n8n
   N8N_SECURE_COOKIE=false n8n start
   ```
3. Open http://localhost:5678 → Import `n8n_workflow.json` → Activate

**Send a query through n8n:**
```bash
curl -X POST http://localhost:5678/webhook/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare transformer and LSTM architectures"}'
```

**Slack integration:** Open the "Post to Slack" node in n8n and replace the URL with your Slack incoming webhook URL.

---

## How the memory system works

The agent uses three memory layers that work together:

**Vector RAG** stores everything the agent learns in ChromaDB. It uses semantic search to find relevant chunks — so if you ask about "reinforcement learning from feedback" it will surface results about RLHF even if the exact phrase isn't in the query.

**Episodic Buffer** keeps the last 10 turns of the current session in RAM. This means the agent remembers what you asked earlier in the same session without re-reading everything from the vector store.

**Summary Cascade** kicks in when the episodic buffer fills up. It compresses the oldest 5 turns into a rolling summary (≤200 words) and keeps that summary in every subsequent context window. This way very long sessions don't blow the token budget.

The **Context Assembler** stacks all three into a single context string in priority order (Summary → Episodic → RAG chunks) while staying inside the remaining token budget.

---

## Token budget

Every query gets a fresh budget. The agent tracks every token spent across all API calls — decomposition, search, synthesis, polish — not just the final response call.

When the remaining budget drops below 1,500 tokens the loop exits cleanly. When it drops below 2,000 the polish step is skipped. You will never get a runaway bill from a single query.

Cost is estimated using published Anthropic rates and logged to `evaluation.md` after every run.

---

## Project structure

```
deep_research_agent/
├── agent.py                  Main orchestrator and agentic loop
├── budget_guard.py           Token counting and budget enforcement
├── context_assembler.py      Fits memory into the token allowance
├── evaluator.py              Logs runs to evaluation.md
├── main.py                   CLI entry point
├── server.py                 FastAPI REST server
├── memory/
│   ├── vector_rag.py         ChromaDB semantic search
│   ├── episodic_buffer.py    Short-term session memory
│   └── summary_cascade.py    Long-term compression
├── n8n_workflow.json         Importable n8n workflow (7 nodes + Slack)
├── docker-compose.yml        Runs n8n + API together in Docker
├── Dockerfile                Container for the API server
├── requirements.txt          Python dependencies
└── evaluation.md             Auto-generated run logs + architecture notes
```

---

## Requirements

- Python 3.9+
- An Anthropic API key (https://console.anthropic.com)
- Node.js + npm (only needed for n8n)
- Docker (only needed for containerised deployment)

Python packages: `anthropic`, `chromadb`, `tiktoken`, `python-dotenv`, `fastapi`, `uvicorn`, `duckduckgo_search`

---

## Business Impact

### The problem this solves

Research is expensive. When a team member needs to understand a topic — a new technology, a competitor, a market trend — they either spend hours reading, or they ask an LLM that hallucinates and has no memory of what was asked last week.

This agent sits between those two extremes. It remembers what your team has already researched, builds on it over time, and gives you a cost-controlled, self-assessed answer every time.

### Concrete value

**Cost control at the query level**
Most LLM deployments have no per-query spending limits. A single complex query with a badly written prompt can cost $2–5. This agent enforces a hard token cap per query — you decide the ceiling before it runs. The `--compare` mode shows you exactly what quality you lose by cutting the budget in half, so you can make an informed call.

**Knowledge compounds over time**
Every answer the agent produces gets stored back into ChromaDB. The third question your team asks about a topic is cheaper and better-answered than the first, because the agent already has context from the previous two. This turns one-off LLM queries into an organisational knowledge base.

**Handles the "I don't know" problem gracefully**
Most agents either hallucinate when they lack data, or return an unhelpful empty response. This agent falls back to live web search automatically, then stores what it finds. If it genuinely cannot help, it says so clearly and suggests where to look — saving time rather than wasting it on a confident wrong answer.

**Non-engineers can trigger it**
The n8n workflow means anyone in the business can POST a question to a webhook and get a structured answer back — no Python, no API keys, no terminal. Hook it up to a Slack bot and the whole team has access to a research agent without engineering involvement.

**Every run is auditable**
`evaluation.md` is a permanent record of every question asked, every token spent, every dollar cost, and a quality score. That is useful for three things: catching regressions if the agent starts performing poorly, justifying the API spend to stakeholders, and identifying which types of questions are expensive so you can pre-load relevant knowledge and cut costs.

### Cost in practice

From live runs during development:

| Query type | Tokens used | Cost | Notes |
|---|---|---|---|
| Focused technical question (RLHF vs DPO) | 9,713 | $0.11 | 19% of 50k budget |
| Broad historical question (evolution of AI) | 13,620 | $0.11 | 27% of 50k budget |
| Real-time question with web search (Champions League) | 11,046 | $0.07 | Web fallback triggered |
| Same question at 15k budget | ~15,000 | ~$0.08 | Answers faster, slightly less depth |

The model split (haiku for search/decomposition, opus for synthesis) reduces per-query cost by roughly 60–70% compared to using Opus for everything, with no noticeable quality drop on the tool-use steps.

---

- `chroma_db/` is not committed to git — it's your local knowledge base. Delete it to start fresh.
- `.env` is not committed — never share your API key.
- The episodic buffer resets on restart by design. Long-term memory lives in ChromaDB.
- All runs are appended to `evaluation.md` — open it any time to review costs and quality scores.

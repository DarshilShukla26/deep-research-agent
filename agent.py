"""Deep Research Agent — main orchestrator.

Architecture (mirrors g3_deep_research_agent_architecture.svg):

  User query
      │
  Router + Budget Guard   ← enforces token_cap per query
      │
  Memory Layer (3 sources retrieved in parallel)
    ├── Vector RAG          (ChromaDB semantic search)
    ├── Episodic Buffer     (recent turns + sub-question log)
    └── Summary Cascade     (compressed history)
      │
  Context Assembler       ← ranks + trims chunks to stay under cap
      │
  Claude (tool use)       ← reasons, calls tools, decomposes
    ├── Sub-question Decomposer  ──► loops back to Memory Layer
    └── Answer Synthesiser       ──► evaluation.md logger ──► Response
"""

from __future__ import annotations

from typing import Optional
import anthropic

from budget_guard import BudgetGuard, TokenBudget
from context_assembler import ContextAssembler
from evaluator import Evaluator
from memory.vector_rag import VectorRAG
from memory.episodic_buffer import EpisodicBuffer
from memory.summary_cascade import SummaryCascade


# ── Tool schemas ────────────────────────────────────────────────────────────
TOOLS: list[dict] = [
    {
        "name": "decompose_question",
        "description": (
            "Break a complex research question into a list of focused "
            "sub-questions that can each be answered independently."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of sub-questions to research.",
                }
            },
            "required": ["sub_questions"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Search the vector knowledge base for relevant information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Semantic search query."},
                "n_results": {
                    "type": "integer",
                    "description": "Number of results (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_to_knowledge",
        "description": "Store a new finding or fact in the vector knowledge base for future retrieval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Text to store."},
                "source":  {"type": "string", "description": "Optional source label."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "synthesize_answer",
        "description": (
            "Produce the final comprehensive answer. "
            "Call this when you have gathered enough information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Complete, well-structured final answer.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence level in the answer.",
                },
                "key_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory sources consulted (e.g. 'vector_rag', 'episodic_buffer').",
                },
            },
            "required": ["answer", "confidence"],
        },
    },
]


class DeepResearchAgent:
    """
    Parameters
    ----------
    token_cap : int
        Maximum tokens (input + output combined) allowed per query.
    model : str
        Claude model to use for all calls.
    chroma_path : str
        Directory for ChromaDB persistence.
    eval_path : str
        Path to the evaluation Markdown log.
    max_iterations : int
        Hard cap on the inner agentic loop to prevent runaway execution.
    """

    def __init__(
        self,
        token_cap: int = 50_000,
        model: str = "claude-opus-4-6",
        fast_model: str = "claude-haiku-4-5",
        chroma_path: str = "./chroma_db",
        eval_path: str = "evaluation.md",
        max_iterations: int = 8,
    ):
        self.model = model
        self.fast_model = fast_model
        self.max_iterations = max_iterations
        self._client = anthropic.Anthropic()

        # ── Memory layers ──────────────────────────────────────────────
        self._rag     = VectorRAG(persist_path=chroma_path)
        self._buf     = EpisodicBuffer(max_turns=10)
        self._cascade = SummaryCascade(client=self._client, model=model)

        # ── Budget + assembly ──────────────────────────────────────────
        self._guard   = BudgetGuard(token_cap=token_cap, model=model)
        self._cascade._guard = self._guard          # share the guard
        self._assembler = ContextAssembler(guard=self._guard)

        # ── Evaluator ──────────────────────────────────────────────────
        self._evaluator = Evaluator(eval_path=eval_path)

    # ── Public API ────────────────────────────────────────────────────────────
    def query(self, user_query: str) -> str:
        """Run a research query end-to-end and return the answer."""
        answer, _, _ = self._run_query(user_query)
        return answer

    def query_full(self, user_query: str) -> dict:
        """Like query() but also returns token usage, cost, and strategy metadata."""
        answer, run_log, budget = self._run_query(user_query)
        return {
            "answer": answer,
            "tokens_input": budget.used_input,
            "tokens_output": budget.used_output,
            "tokens_total": budget.used_total,
            "cost_usd": budget.cost_usd(self.model),
            "memory_strategies": run_log["memory_strategies"],
            "sub_questions": run_log["sub_questions"],
            "iterations": run_log["iterations"],
            "budget_utilisation_pct": round(
                budget.used_total / budget.total_cap * 100, 1
            ) if budget.total_cap else 0.0,
        }

    def _run_query(self, user_query: str) -> tuple:
        """Shared implementation — returns (answer, run_log, budget)."""
        budget = self._guard.new_budget()
        run_log: dict = {
            "query": user_query,
            "memory_strategies": [],
            "sub_questions": [],
            "iterations": 0,
        }

        try:
            result = self._research_loop(user_query, budget, run_log)
            result = self._polish_answer(result, user_query, budget)
            score = self._self_score(user_query, result, budget)
            run_log["self_score"] = score
        except Exception as exc:
            result = f"[Agent error: {exc}]"
        finally:
            self._evaluator.log_run(run_log, budget, self.model)

        return result, run_log, budget

    def ingest(self, text: str, metadata: Optional[dict] = None) -> str:
        """Pre-load external knowledge into the vector store."""
        return self._rag.add(text, metadata)

    # ── Internal loop ─────────────────────────────────────────────────────────
    def _research_loop(
        self, query: str, budget: TokenBudget, run_log: dict
    ) -> str:

        messages: list[dict] = []
        final_answer: Optional[str] = None

        for iteration in range(self.max_iterations):
            if budget.is_exhausted():
                break

            run_log["iterations"] = iteration + 1

            # ── Step 1: Retrieve from all memory layers ────────────────
            rag_chunks = self._rag.search(query, n_results=6)
            episodic   = self._buf.get_recent()
            summary    = self._cascade.get_summary()

            context, strategies = self._assembler.assemble(
                budget, rag_chunks, episodic, summary
            )
            for s in strategies:
                if s not in run_log["memory_strategies"]:
                    run_log["memory_strategies"].append(s)

            # ── Step 2: Build system prompt ────────────────────────────
            system = self._build_system(context, budget)

            # ── Step 3: First turn or extend existing conversation ─────
            if not messages:
                messages = [{"role": "user", "content": query}]

            # ── Step 4: Safety check ───────────────────────────────────
            if budget.remaining < 1_500:
                break

            max_out = min(4_096, budget.remaining // 2)

            # ── Step 5: Call Claude ────────────────────────────────────
            response = self._client.messages.create(
                model=self.fast_model,
                max_tokens=max_out,
                system=system,
                messages=messages,
                tools=TOOLS,
            )
            self._guard.record(budget, response.usage.input_tokens, response.usage.output_tokens)

            # ── Step 6: Append assistant turn ─────────────────────────
            messages.append({"role": "assistant", "content": response.content})

            # ── Step 7: Process content blocks ────────────────────────
            tool_results: list[dict] = []
            done = False

            for block in response.content:
                if block.type != "tool_use":
                    continue

                result_content, stop = self._dispatch_tool(
                    block, query, budget, run_log
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })
                if stop:
                    final_answer = result_content
                    done = True
                    break

            if done:
                break

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            elif response.stop_reason == "end_turn":
                # Claude finished without a tool call — extract text
                for block in response.content:
                    if hasattr(block, "text"):
                        final_answer = block.text
                        break
                break

        # ── Post-loop: update episodic buffer ──────────────────────────
        answer_to_store = final_answer or "No answer synthesized within budget."
        self._buf.add_turn(query, answer_to_store)

        # Trigger summary cascade if buffer overflows
        if self._buf.is_full():
            self._cascade.compress(self._buf, budget)
            if "summary_cascade" not in run_log["memory_strategies"]:
                run_log["memory_strategies"].append("summary_cascade")

        return answer_to_store

    def _polish_answer(self, raw_answer: str, query: str, budget: TokenBudget) -> str:
        """Use the full model to rewrite the raw answer with better structure."""
        if not raw_answer or raw_answer.startswith("[Agent error"):
            return raw_answer
        if budget.remaining < 2_000:
            return raw_answer  # not enough budget to polish

        prompt = (
            f"Original question: {query}\n\n"
            f"Draft answer:\n{raw_answer}\n\n"
            "Rewrite this answer with:\n"
            "• ## section headers for each major topic\n"
            "• **bold** for key terms\n"
            "• Bullet points or numbered lists where appropriate\n"
            "• A ## Summary section at the end with 3-5 key takeaways\n"
            "Keep all the information — only improve the formatting and clarity."
        )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=min(2_048, budget.remaining // 2),
            messages=[{"role": "user", "content": prompt}],
        )
        self._guard.record(budget, resp.usage.input_tokens, resp.usage.output_tokens)
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return raw_answer

    def _self_score(self, query: str, answer: str, budget: TokenBudget) -> dict:
        """Rate the answer quality using the fast model. Returns score dict."""
        if not answer or answer.startswith("[Agent error") or budget.remaining < 1_000:
            return {}

        prompt = (
            f"Question: {query}\n\nAnswer:\n{answer}\n\n"
            "Rate this answer on each dimension from 1 (poor) to 5 (excellent):\n"
            "1. Completeness — does it fully address all parts of the question?\n"
            "2. Clarity — is it well-structured and easy to read?\n"
            "3. Accuracy — does it appear factually sound and well-reasoned?\n\n"
            "Reply with ONLY a JSON object, no explanation:\n"
            '{"completeness": <1-5>, "clarity": <1-5>, "accuracy": <1-5>, "overall": <1-5>, "note": "<one sentence>"}'
        )
        try:
            resp = self._client.messages.create(
                model=self.fast_model,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            self._guard.record(budget, resp.usage.input_tokens, resp.usage.output_tokens)
            import json, re
            for block in resp.content:
                if hasattr(block, "text"):
                    match = re.search(r'\{.*\}', block.text, re.DOTALL)
                    if match:
                        return json.loads(match.group())
        except Exception:
            pass
        return {}

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _build_system(self, context: str, budget: TokenBudget) -> str:
        budget_line = (
            f"Token budget: {budget.remaining:,} remaining / {budget.total_cap:,} total."
        )
        base = (
            "You are a deep research agent with a three-layer memory system.\n\n"
            "Available tools:\n"
            "• decompose_question — split a complex question into sub-questions\n"
            "• search_knowledge   — semantic search in the vector knowledge base\n"
            "• add_to_knowledge   — persist a new finding for future retrieval\n"
            "• synthesize_answer  — emit the final answer (ends the loop)\n\n"
            f"{budget_line}\n\n"
            "ANSWER FORMATTING RULES (strictly follow these):\n"
            "• Use clear Markdown: ## for main sections, ### for sub-sections\n"
            "• Use **bold** for key terms and important concepts\n"
            "• Use bullet points (•) or numbered lists for comparisons and steps\n"
            "• Add a ## Summary section at the end with 3-5 key takeaways\n"
            "• Keep paragraphs short (3-4 sentences max)\n"
            "• Never return a wall of unbroken text\n"
            "• When you have enough information, call synthesize_answer.\n"
        )
        if context:
            base += f"\n\n{context}"
        return base

    def _dispatch_tool(
        self,
        block,
        original_query: str,
        budget: TokenBudget,
        run_log: dict,
    ) -> tuple[str, bool]:
        """
        Execute a tool call.

        Returns
        -------
        (result_content, should_stop)
        """
        name = block.name
        inp  = block.input

        if name == "decompose_question":
            sub_qs: list[str] = inp.get("sub_questions", [])
            run_log["sub_questions"].extend(sub_qs)
            for sq in sub_qs:
                self._buf.add_sub_question(sq)
                # Eagerly search each sub-question and ingest results
                hits = self._rag.search(sq, n_results=3)
                for hit in hits:
                    # Already in store; no-op (add is idempotent)
                    pass
            return (
                "Sub-questions registered: " + "; ".join(sub_qs),
                False,
            )

        if name == "search_knowledge":
            results = self._rag.search(
                inp["query"], n_results=inp.get("n_results", 5)
            )
            if results:
                return "\n---\n".join(results), False
            return "No results found in knowledge base.", False

        if name == "add_to_knowledge":
            meta = {"source": inp.get("source", "agent")}
            self._rag.add(inp["content"], meta)
            return "Stored successfully.", False

        if name == "synthesize_answer":
            answer     = inp.get("answer", "")
            confidence = inp.get("confidence", "medium")
            sources    = inp.get("key_sources", [])
            formatted  = answer
            if confidence or sources:
                formatted += f"\n\n[Confidence: {confidence}"
                if sources:
                    formatted += f" | Sources: {', '.join(sources)}"
                formatted += "]"
            return formatted, True   # stop the loop

        return f"Unknown tool '{name}'.", False

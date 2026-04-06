"""Context assembler — ranks and trims memory chunks to fit the token budget."""

from __future__ import annotations

from budget_guard import BudgetGuard, TokenBudget


class ContextAssembler:
    """Combines retrieved memory slices into a single context string
    while staying within the remaining token budget."""

    # Reserve this many tokens for the actual Claude response
    RESPONSE_RESERVE = 4_096

    def __init__(self, guard: BudgetGuard):
        self._guard = guard

    def assemble(
        self,
        budget: TokenBudget,
        rag_chunks: list[str],
        episodic_text: str,
        summary_text: str,
    ) -> tuple[str, list[str]]:
        """
        Returns:
            (assembled_context, strategies_used)

        Strategies used is a subset of ["vector_rag", "episodic_buffer", "summary_cascade"].
        """
        allowance = budget.remaining - self.RESPONSE_RESERVE
        if allowance <= 0:
            return "", []

        parts: list[str] = []
        strategies: list[str] = []

        # 1. Historical summary (most compressed — goes first)
        if summary_text:
            header = "## Historical Summary\n"
            token_cost = self._guard.count(header + summary_text)
            if token_cost <= allowance:
                parts.append(header + summary_text)
                strategies.append("summary_cascade")
                allowance -= token_cost

        # 2. Recent episodic turns
        if episodic_text:
            header = "## Recent Research Turns\n"
            token_cost = self._guard.count(header + episodic_text)
            if token_cost <= allowance:
                parts.append(header + episodic_text)
                strategies.append("episodic_buffer")
                allowance -= token_cost

        # 3. Vector RAG chunks — greedily include until budget is tight
        included_chunks: list[str] = []
        for chunk in rag_chunks:
            cost = self._guard.count(chunk) + 4   # "+4" for separator
            if cost <= allowance:
                included_chunks.append(chunk)
                allowance -= cost
            # Don't break — a later shorter chunk might still fit
        if included_chunks:
            parts.append("## Relevant Knowledge\n" + "\n---\n".join(included_chunks))
            strategies.append("vector_rag")

        return "\n\n".join(parts), strategies

"""Summary cascade — compresses old turns when the episodic buffer overflows."""

from __future__ import annotations

from typing import Optional
import anthropic

from budget_guard import TokenBudget, BudgetGuard
from memory.episodic_buffer import EpisodicBuffer, Turn


class SummaryCascade:
    """Uses Claude to compress old episodic turns into a rolling summary."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-opus-4-6",
        guard: Optional[BudgetGuard] = None,
    ):
        self._client = client
        self._model = model
        self._guard = guard
        self._summary: str = ""

    # ------------------------------------------------------------------ read
    def get_summary(self) -> str:
        return self._summary

    # ------------------------------------------------------------------ write
    def compress(self, buffer: EpisodicBuffer, budget: TokenBudget) -> None:
        """Drain old turns from the buffer and fold them into the rolling summary."""
        old_turns: list[Turn] = buffer.drain_oldest(n=5)
        if not old_turns:
            return

        turns_text = "\n".join(
            f"Q: {t.query}\nA: {t.answer[:600]}" for t in old_turns
        )
        prior = f"Previous summary:\n{self._summary}\n\n" if self._summary else ""

        prompt = (
            f"{prior}"
            f"New conversation turns to integrate:\n{turns_text}\n\n"
            "Write a concise rolling summary (≤200 words) that preserves "
            "key facts, findings, and sub-questions already answered."
        )

        # Estimate cost; skip if we can't afford it
        if self._guard and not self._guard.fits(budget, self._guard.count(prompt) + 300):
            # Fall back: just concatenate
            self._summary = (self._summary + "\n" + turns_text)[:1200]
            return

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        if self._guard:
            self._guard.record(budget, resp.usage.input_tokens, resp.usage.output_tokens)

        for block in resp.content:
            if block.type == "text":
                self._summary = block.text
                break

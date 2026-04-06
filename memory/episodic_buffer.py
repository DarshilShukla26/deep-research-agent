"""Episodic buffer — keeps recent turns + sub-question log in memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    query: str
    answer: str


class EpisodicBuffer:
    """Fixed-capacity ring of recent (query, answer) turns and sub-question log."""

    def __init__(self, max_turns: int = 10):
        self._max = max_turns
        self._turns: list[Turn] = []
        self._sub_questions: list[str] = []

    # ------------------------------------------------------------------ write
    def add_turn(self, query: str, answer: str) -> None:
        self._turns.append(Turn(query=query, answer=answer))
        if len(self._turns) > self._max:
            self._turns.pop(0)          # drop oldest

    def add_sub_question(self, question: str) -> None:
        self._sub_questions.append(question)

    # ------------------------------------------------------------------ read
    def get_recent(self, n: Optional[int] = None) -> str:
        turns = self._turns[-n:] if n else self._turns
        if not turns:
            return ""
        lines = []
        for t in turns:
            lines.append(f"Q: {t.query}")
            lines.append(f"A: {t.answer[:400]}")
            lines.append("---")
        return "\n".join(lines)

    def get_sub_questions(self) -> list[str]:
        return list(self._sub_questions)

    def is_full(self) -> bool:
        return len(self._turns) >= self._max

    def drain_oldest(self, n: int = 5) -> list[Turn]:
        """Remove and return the n oldest turns for compression."""
        drained = self._turns[:n]
        self._turns = self._turns[n:]
        return drained

    def __len__(self) -> int:
        return len(self._turns)

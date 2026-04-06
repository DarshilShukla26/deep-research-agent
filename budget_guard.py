"""Token budget guard — enforces a per-query token cap."""

from dataclasses import dataclass, field
import tiktoken


# Approximate cost per 1M tokens (USD) for claude-opus-4-6
COST_PER_1M = {
    "claude-opus-4-6":   {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":  {"input": 1.00,  "output":  5.00},
}


@dataclass
class TokenBudget:
    total_cap: int
    used_input: int = 0
    used_output: int = 0

    @property
    def used_total(self) -> int:
        return self.used_input + self.used_output

    @property
    def remaining(self) -> int:
        return max(0, self.total_cap - self.used_total)

    def is_exhausted(self) -> bool:
        return self.remaining == 0

    def cost_usd(self, model: str) -> float:
        rates = COST_PER_1M.get(model, COST_PER_1M["claude-opus-4-6"])
        return (
            self.used_input  / 1_000_000 * rates["input"] +
            self.used_output / 1_000_000 * rates["output"]
        )


class BudgetGuard:
    """Counts tokens and enforces a per-query cap."""

    def __init__(self, token_cap: int = 50_000, model: str = "claude-opus-4-6"):
        self.token_cap = token_cap
        self.model = model
        # cl100k_base is a reasonable approximation for Claude models
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def new_budget(self) -> TokenBudget:
        return TokenBudget(total_cap=self.token_cap)

    def fits(self, budget: TokenBudget, estimated: int) -> bool:
        """Return True if `estimated` more tokens can still be spent."""
        return budget.remaining >= estimated

    def record(self, budget: TokenBudget, input_tokens: int, output_tokens: int) -> None:
        budget.used_input  += input_tokens
        budget.used_output += output_tokens

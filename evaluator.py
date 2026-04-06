"""Evaluation logger — appends one run record to evaluation.md."""

import os
from datetime import datetime

from budget_guard import TokenBudget, COST_PER_1M


class Evaluator:
    """Appends a structured Markdown run record to evaluation.md."""

    def __init__(self, eval_path: str = "evaluation.md"):
        self._path = eval_path
        self._ensure_header()

    # ------------------------------------------------------------------ private
    def _ensure_header(self) -> None:
        if not os.path.exists(self._path):
            with open(self._path, "w") as f:
                f.write("# Deep Research Agent — Evaluation Log\n\n")
                f.write("Each section below is one run.\n\n---\n\n")

    # ------------------------------------------------------------------ public
    def log_run(self, run_log: dict, budget: TokenBudget, model: str) -> None:
        """Append a run record to evaluation.md."""
        rates = COST_PER_1M.get(model, COST_PER_1M["claude-opus-4-6"])
        cost_usd = budget.cost_usd(model)

        lines = [
            f"## Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"**Query:** {run_log.get('query', '—')}",
            "",
            "### Token Usage",
            f"| Field | Value |",
            f"|---|---|",
            f"| Model | `{model}` |",
            f"| Input tokens | {budget.used_input:,} |",
            f"| Output tokens | {budget.used_output:,} |",
            f"| Total tokens | {budget.used_total:,} |",
            f"| Token cap | {budget.total_cap:,} |",
            f"| Utilisation | {budget.used_total / budget.total_cap * 100:.1f}% |",
            "",
            "### Cost Estimate",
            f"| Component | Tokens | Rate ($/1M) | Cost ($) |",
            f"|---|---|---|---|",
            f"| Input | {budget.used_input:,} | {rates['input']:.2f} | "
            f"{budget.used_input / 1_000_000 * rates['input']:.6f} |",
            f"| Output | {budget.used_output:,} | {rates['output']:.2f} | "
            f"{budget.used_output / 1_000_000 * rates['output']:.6f} |",
            f"| **Total** | | | **{cost_usd:.6f}** |",
            "",
            "### Memory Strategies Triggered",
        ]

        strategies = run_log.get("memory_strategies", [])
        if strategies:
            for s in strategies:
                lines.append(f"- {s}")
        else:
            lines.append("- *(none)*")

        sub_qs = run_log.get("sub_questions", [])
        if sub_qs:
            lines.append("")
            lines.append("### Sub-Questions Decomposed")
            for q in sub_qs:
                lines.append(f"- {q}")

        lines.append("")
        lines.append(f"**Iterations:** {run_log.get('iterations', 0)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        with open(self._path, "a") as f:
            f.write("\n".join(lines) + "\n")

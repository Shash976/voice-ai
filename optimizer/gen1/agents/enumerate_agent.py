"""enumerate_agent.py — exhaustive grid search (the correct tool for small spaces).

This is NOT a learning agent and makes no pretence of being one.  The sim search
space (search_space.yaml) is 5×3×3 = 45 configs — small, deterministic, and fully
enumerable.  On a space that size there is nothing for a "smart" search to exploit
(see docs/04_optimizer.md, "the honest finding"): the right answer is to evaluate
every config exactly once and report the true global optimum, with zero sampling
variance.

It satisfies the BaseAgent interface so run_optimizer can drive it like any other
strategy: suggest() walks the full Cartesian product in a fixed order, then (once
exhausted) keeps returning the best config seen so it degrades gracefully if asked
for more trials than the space holds.  Use the `space_size` property to size a run
so it covers the grid exactly.
"""

from __future__ import annotations

from .base_agent import BaseAgent


class EnumerateAgent(BaseAgent):
    """Deterministic exhaustive sweep over the entire search space."""

    def __init__(self, search_space: dict) -> None:
        super().__init__(search_space)
        self._configs = self._all_configs()   # full Cartesian product, fixed order
        self._idx = 0
        self._best_config: dict | None = None
        self._best_reward = float("-inf")

    @property
    def space_size(self) -> int:
        """Number of distinct configs in the grid — use this to size the run."""
        return len(self._configs)

    def suggest(self, state: dict, history: list[dict]) -> dict:
        if self._idx < len(self._configs):
            cfg = self._configs[self._idx]
            self._idx += 1
            return cfg
        # Grid exhausted: re-propose the best seen so extra trials are harmless.
        return dict(self._best_config or self._configs[-1])

    def update(self, config: dict, reward: float, info: dict) -> None:
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_config = dict(config)

    def warm_start(self, history: list[dict]) -> None:
        """On --resume, skip configs already in the results file and seed the best."""
        seen = {str(sorted(r["config"].items())) for r in history if "config" in r}
        for r in history:
            self.update(r.get("config", {}), r.get("reward", float("-inf")), r)
        # Advance the cursor past configs we've already evaluated.
        while (self._idx < len(self._configs)
               and str(sorted(self._configs[self._idx].items())) in seen):
            self._idx += 1

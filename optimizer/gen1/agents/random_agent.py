"""random_agent.py — uniform random search baseline.

Samples each config independently and uniformly from the search space.
Useful as a sanity-check baseline to compare other agents against.
No external dependencies.
"""

from __future__ import annotations

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Pure random search: every suggestion is an independent uniform sample."""

    def suggest(self, state: dict, history: list[dict]) -> dict:
        return self.random_config()

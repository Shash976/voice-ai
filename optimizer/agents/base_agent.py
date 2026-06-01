"""base_agent.py — abstract base class for all optimizer agents."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    All agents share this interface:

        config = agent.suggest(state, history)   # propose next config to evaluate
        agent.update(config, reward, info)        # record the outcome (optional)

    Agents must not call the simulator directly; OptEnv.step() does that.
    """

    def __init__(self, search_space: dict) -> None:
        self.search_space = search_space
        self._rng = random.Random()

    @abstractmethod
    def suggest(self, state: dict, history: list[dict]) -> dict:
        """Return the next config dict to evaluate."""

    def update(self, config: dict, reward: float, info: dict) -> None:
        """Called after OptEnv.step() completes. Override to update internal policy."""

    def warm_start(self, history: list[dict]) -> None:
        """
        Replay previously observed (config, reward) pairs into the agent's
        internal state.  Called when --resume loads an existing results file
        so the agent continues learning from where it left off rather than
        starting cold.  Default: no-op (stateless agents don't need it).
        """

    # ── Helpers ───────────────────────────────────────────────────────────────

    def random_config(self) -> dict:
        return {
            name: self._rng.choice(spec["choices"])
            for name, spec in self.search_space.items()
        }

    def _mutate_one(self, config: dict, name: str) -> dict:
        """Return a copy of config with one parameter changed to a different value."""
        spec    = self.search_space[name]
        others  = [c for c in spec["choices"] if c != config[name]]
        child   = dict(config)
        child[name] = self._rng.choice(others) if others else config[name]
        return child

    def _all_configs(self) -> list[dict]:
        """Enumerate every combination in the search space (only for tiny spaces)."""
        from itertools import product
        names  = list(self.search_space.keys())
        combos = list(product(*[s["choices"] for s in self.search_space.values()]))
        return [dict(zip(names, combo)) for combo in combos]

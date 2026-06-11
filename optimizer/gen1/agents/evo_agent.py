"""evo_agent.py — (mu + lambda) evolutionary optimizer for categorical spaces.

Algorithm:
  1. First mu trials: sample randomly to seed the population.
  2. Each subsequent trial:
     a. Select the top-mu elites from all evaluated configs.
     b. Pick one elite at random as the parent.
     c. Produce a child by mutating each parameter independently
        with probability p_mutate.
  3. Occasionally inject a random immigrant (with prob p_random) to
     prevent premature convergence.

No external dependencies — pure Python stdlib.
"""

from __future__ import annotations

from .base_agent import BaseAgent


class EvoAgent(BaseAgent):
    """
    (μ + λ) evolutionary strategy for discrete/categorical search spaces.

    Parameters
    ----------
    mu        : elite population size (survivors per generation)
    p_mutate  : per-parameter mutation probability
    p_random  : probability of injecting a fully random config each trial
    """

    def __init__(
        self,
        search_space: dict,
        mu: int = 5,
        p_mutate: float = 0.30,
        p_random: float = 0.10,
    ) -> None:
        super().__init__(search_space)
        self.mu       = mu
        self.p_mutate = p_mutate
        self.p_random = p_random
        # (config, reward) pairs for all evaluated configs
        self._evaluated: list[tuple[dict, float]] = []

    def suggest(self, state: dict, history: list[dict]) -> dict:
        # ── Phase 1: seed with random configs ──
        if len(self._evaluated) < self.mu:
            return self.random_config()

        # ── Random immigrant ──
        if self._rng.random() < self.p_random:
            return self.random_config()

        # ── Select elites and mutate ──
        elites = sorted(self._evaluated, key=lambda x: x[1], reverse=True)[: self.mu]
        parent_cfg, _ = self._rng.choice(elites)

        child = dict(parent_cfg)
        for name, spec in self.search_space.items():
            if self._rng.random() < self.p_mutate:
                others = [c for c in spec["choices"] if c != child[name]]
                if others:
                    child[name] = self._rng.choice(others)

        return child

    def update(self, config: dict, reward: float, info: dict) -> None:
        self._evaluated.append((config, reward))
        # Bound memory: keep at most 3*mu records
        cap = max(self.mu * 3, 30)
        if len(self._evaluated) > cap:
            self._evaluated.sort(key=lambda x: x[1], reverse=True)
            self._evaluated = self._evaluated[: self.mu * 2]

    def warm_start(self, history: list[dict]) -> None:
        """Replay historical (config, reward) pairs into the elite population."""
        for record in history:
            config = record.get("config") or {}
            reward = record.get("reward", 0.0)
            if config:
                self._evaluated.append((config, reward))

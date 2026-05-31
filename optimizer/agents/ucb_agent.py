"""ucb_agent.py — Factored UCB1 bandit for multi-dimensional categorical spaces.

Each parameter dimension is treated as an independent multi-armed bandit.
For dimension d with choices [v0, v1, ...], UCB1 selects the value that
maximises:  mean_reward[d][v]  +  c * sqrt(ln(N) / n[d][v])

where N = total trials and n[d][v] = how many times value v was chosen for d.

This is a simplification (assumes parameter independence), but it works well
in practice for small search spaces and is completely dependency-free.

The exploration constant c controls the exploitation/exploration tradeoff:
  c → 0  : pure greedy exploitation
  c = √2 : classic UCB1 theoretical optimum (for rewards in [0, 1])
  c > √2 : more exploration

For our reward range (roughly −100 … +3), rewards are shifted and scaled
internally so UCB arithmetic stays numerically stable.
"""

from __future__ import annotations

import math

from .base_agent import BaseAgent


class UCBAgent(BaseAgent):
    """
    Factored UCB1 bandit.

    Parameters
    ----------
    c : float
        Exploration constant (default √2 ≈ 1.414).
    """

    def __init__(self, search_space: dict, c: float = math.sqrt(2)) -> None:
        super().__init__(search_space)
        self.c = c
        # Per-dimension, per-value reward history
        self._rewards: dict[str, dict] = {
            name: {v: [] for v in spec["choices"]}
            for name, spec in search_space.items()
        }
        self._total_trials: int = 0

    def suggest(self, state: dict, history: list[dict]) -> dict:
        config: dict = {}

        for name, spec in self.search_space.items():
            # Any unvisited choice takes priority (ensure full coverage first)
            unvisited = [v for v in spec["choices"] if not self._rewards[name][v]]
            if unvisited:
                config[name] = self._rng.choice(unvisited)
                continue

            # UCB1 selection across visited choices
            best_val = None
            best_ucb = float("-inf")
            log_N    = math.log(max(self._total_trials, 1))

            for v in spec["choices"]:
                rew_list = self._rewards[name][v]
                n   = len(rew_list)
                mu  = self._mean_reward(rew_list)
                ucb = mu + self.c * math.sqrt(log_N / n)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_val = v

            config[name] = best_val

        return config

    def update(self, config: dict, reward: float, info: dict) -> None:
        self._total_trials += 1
        for name in self.search_space:
            v = config.get(name)
            if v is not None and v in self._rewards[name]:
                self._rewards[name][v].append(reward)

    # Shift rewards so that the minimum observed ≥ 0 (UCB1 requires non-negative).
    def _mean_reward(self, rew_list: list[float]) -> float:
        if not rew_list:
            return 0.0
        # Shift by global min seen across all dimensions
        all_vals = [r for dim in self._rewards.values() for lst in dim.values() for r in lst]
        shift = -min(all_vals) if all_vals else 0.0
        return sum(r + shift for r in rew_list) / len(rew_list)

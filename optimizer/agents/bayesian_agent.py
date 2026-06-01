"""bayesian_agent.py — Optuna TPE Bayesian optimizer.

Wraps optuna.Study so it fits the BaseAgent interface.
Requires: pip install --user optuna

TPE (Tree-structured Parzen Estimator) builds probabilistic models of
p(reward | config) and samples from the promising region of the space.
It outperforms random search after ~10–20 trials and handles categorical
parameters natively — ideal for our discrete design space.
"""

from __future__ import annotations

from .base_agent import BaseAgent


class BayesianAgent(BaseAgent):
    """
    Optuna TPE-based Bayesian optimizer.

    Parameters
    ----------
    n_startup_trials : int
        Number of random warm-up trials before TPE kicks in (default 5).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        search_space: dict,
        n_startup_trials: int = 5,
        seed: int | None = None,
    ) -> None:
        super().__init__(search_space)
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "BayesianAgent requires optuna.  Install with:\n"
                "  pip install --user optuna"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            seed=seed,
        )
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._pending: "optuna.Trial | None" = None

    def suggest(self, state: dict, history: list[dict]) -> dict:
        self._pending = self._study.ask()
        config: dict = {}
        for name, spec in self.search_space.items():
            config[name] = self._pending.suggest_categorical(name, spec["choices"])
        return config

    def update(self, config: dict, reward: float, info: dict) -> None:
        if self._pending is not None:
            self._study.tell(self._pending, reward)
            self._pending = None

    def warm_start(self, history: list[dict]) -> None:
        """Inject historical trials into the Optuna study so TPE uses them."""
        try:
            import optuna
        except ImportError:
            return

        for record in history:
            config = record.get("config") or {}
            reward = record.get("reward", 0.0)
            if not config:
                continue
            params = {
                name: config[name]
                for name in self.search_space
                if name in config
            }
            if not params:
                continue
            distributions = {
                name: optuna.distributions.CategoricalDistribution(
                    self.search_space[name]["choices"]
                )
                for name in params
            }
            trial = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=reward,
            )
            self._study.add_trial(trial)

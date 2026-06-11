"""ucb_agent.py — Factored UCB1 bandit for multi-dimensional categorical spaces.

Each parameter dimension is treated as an independent multi-armed bandit.
For dimension d with choices [v0, v1, …], UCB1 selects:

    argmax_v  mu(d, v)  +  c · √( ln(N) / n(d, v) )

where N = total trials so far and n(d, v) = pulls of value v on dimension d.

Reward normalisation
--------------------
UCB1's theoretical guarantees require rewards in [0, 1].  We use fixed bounds
derived from the reward formula rather than a running minimum (which is
non-stationary and causes the entire history to silently re-interpret itself
every time a new worst result arrives).

These bounds were RE-DERIVED on 2026-06-02 after the frequency-coupled reward
rewrite (real_speedup, max_speedup=576).  The reward no longer reaches −60: the
overflow penalty is −50×(1−accuracy) = −50×0.266 = −13.3 for the synthetic
int16 case (accuracy 0.734), and the realistic extremes over the full remaining
space (5 lanes × 3 acc × 3 clk) are:

    worst ≈ −10.55  (lanes=1, acc=16, clk=20: acc penalty −13.3 + norm_spd ≈
                     −0.something + small area/power; floor not hit as the
                     overflow case still clears 10× real_speedup)
    best  ≈  +4.01  (lanes=4, acc=24, clk=5: norm_spd 0.87, low area/power)

    REWARD_LO = −12.0   (just below the worst realistic reward, −10.55)
    REWARD_HI =   4.5   (just above the best realistic reward, +4.01)

Check: (4.01 − (−12.0)) / 16.5 = 0.970 < 1 (optimum does NOT clamp to 1.0);
       (−10.55 − (−12.0)) / 16.5 = 0.088 > 0 (worst stays inside [0, 1]).

Independence assumption
-----------------------
Factoring assumes parameters are independent.  This is wrong for coupled
rewards (area = lanes × acc_width), but tolerable in practice because the
dominant signal (acc_width=16 → −50 overflow) is separable, and the
exploration bonus provides enough coverage to discover the joint optimum.
A Thompson-sampling or GP-UCB alternative would handle coupling exactly.

Exploration constant
--------------------
c = √2 is the standard UCB1 choice.  With rewards normalised to [0, 1] and
the bonus on the order of 0.3–1.5 at typical trial counts, exploration is
meaningful (not decorative).
"""

from __future__ import annotations

import math

from .base_agent import BaseAgent

# Fixed normalisation bounds — derived from reward formula extremes (2026-06-02).
# NEVER use a running minimum: it shifts historical means when a new low arrives.
# See module docstring for the derivation: realistic reward range ≈ [−10.55, +4.01].
_REWARD_LO: float = -12.0
_REWARD_HI: float =   4.5
_REWARD_RNG: float = _REWARD_HI - _REWARD_LO   # 16.5


class UCBAgent(BaseAgent):
    """
    Factored UCB1 bandit.

    Parameters
    ----------
    c : float
        Exploration constant.  Default √2 (classic UCB1 value for [0,1] rewards).
    reward_bounds : tuple[float, float] | None
        ``(lo, hi)`` used for reward normalisation.  When *None* the module-level
        defaults ``(_REWARD_LO, _REWARD_HI)`` = ``(−12.0, 4.5)`` are used, which
        cover the 45-config behavioral track.  Pass ``(-100.0, 4.5)`` for the
        physical / cascade tracks whose penalty ladder reaches −100; otherwise
        −40/−60/−80/−100 all clamp to 0.0 and the penalty structure is invisible.
    """

    def __init__(self, search_space: dict, c: float = math.sqrt(2),
                 reward_bounds: tuple | None = None) -> None:
        super().__init__(search_space)
        self.c = c
        if reward_bounds is not None:
            self._reward_lo  = float(reward_bounds[0])
            self._reward_hi  = float(reward_bounds[1])
        else:
            self._reward_lo  = _REWARD_LO
            self._reward_hi  = _REWARD_HI
        self._reward_rng = self._reward_hi - self._reward_lo
        self._rewards: dict[str, dict] = {
            name: {v: [] for v in spec["choices"]}
            for name, spec in search_space.items()
        }
        self._total_trials: int = 0

    def suggest(self, state: dict, history: list[dict]) -> dict:
        config: dict = {}
        for name, spec in self.search_space.items():
            # First ensure every choice has been tried at least once
            unvisited = [v for v in spec["choices"] if not self._rewards[name][v]]
            if unvisited:
                config[name] = self._rng.choice(unvisited)
                continue

            # UCB1 over visited choices using stable normalised rewards
            best_val = None
            best_ucb = float("-inf")
            log_N    = math.log(max(self._total_trials, 1))

            for v in spec["choices"]:
                rew_list = self._rewards[name][v]
                n   = len(rew_list)
                mu  = self._normalised_mean(rew_list)
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

    def warm_start(self, history: list[dict]) -> None:
        """Replay historical (config, reward) pairs into the bandit's tables."""
        for record in history:
            config = record.get("config") or {}
            reward = record.get("reward", 0.0)
            if not config:
                continue
            self._total_trials += 1
            for name in self.search_space:
                v = config.get(name)
                if v is not None and v in self._rewards[name]:
                    self._rewards[name][v].append(reward)

    def _normalised_mean(self, rew_list: list[float]) -> float:
        """Map raw rewards to [0, 1] using instance bounds, then average."""
        if not rew_list:
            return 0.0
        lo, rng = self._reward_lo, self._reward_rng
        return sum(
            max(0.0, min(1.0, (r - lo) / rng))
            for r in rew_list
        ) / len(rew_list)

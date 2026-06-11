"""surrogate.py — multi-fidelity surrogate that predicts final (F3) physical
metrics from a config x, optionally conditioned on cheaper F2 observables.

Architecture
------------
Per-metric ensemble of three GradientBoostingRegressor instances trained on
three quantile loss values (q=0.16, q=0.50, q=0.84), following the quantile-
regression ensemble (QRE) approach.  Using quantile GBT rather than a bagged
ensemble avoids the n_estimators×3 memory cost while giving calibrated interval
estimates: sigma ≈ (q84 − q16) / 2, mu = q50 prediction.

Features
--------
Config features (always present):
  - log2(lanes)        — log scale because area/cycles grow sub-linearly
  - acc_w              — raw; three discrete values {16,24,32}
  - clk_ns             — continuous clock constraint (the single biggest driver
                         of Fmax and power, per Phase 5 data)
  - abc_area_flag      — 1 if abc_recipe == 'area', 0 otherwise
  - platform_flag      — 1 if platform == 'asap7', 0 otherwise
  - rtl_hash_cat       — integer-encoded RTL hash (context for transfer; one
                         value per RTL version in the corpus)

Conditional F2 observables (present or missing — handled with indicator cols):
  - proxy_area_um2     — synth cell area × 1.35 inflation estimate
  - proxy_wns_ns       — pre-layout WNS from fast STA (sign of proxy timing)
  - ff_count           — sequential cell count from finish report JSON
  - cell_count         — total standard-cell count from finish report JSON
  - logic_levels       — not available from the current build artifacts; field
                         accepted but left always-missing (indicator = 0)

For each obs column, a paired "obs_<col>_present" indicator is appended so
a single model handles both the x-only and x+obs prediction modes.

Small-n fallback
----------------
If n < 10 training rows, fit is skipped.  predict() returns (mean, large_sigma)
where large_sigma = 2 × stddev of observed values (or a hard fallback if n==0).
This is documented behaviour, not a crash.

Composite-reward propagation
-----------------------------
predict_reward_stats() propagates the per-metric predictions through a crude
first-order uncertainty propagation of the physical_reward formula.  It is
intentionally approximate — its main use is as a UCB acquisition signal, not
a calibrated confidence interval.

Saving / loading
----------------
joblib.dump/load; the file stores (model dict, metadata).

Usage
-----
    s = Surrogate(seed=42)
    diag = s.fit(rows)          # rows: list[dict] in results_physical / funnel format
    mu_s, sig_s = s.predict({"mac_lanes": 4, "accumulator_width": 24,
                              "clock_period_ns": 5.0})["area_um2"]
    s.save("surrogate_n45.joblib")
    s2 = Surrogate.load("surrogate_n45.joblib")
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# ── Constants ──────────────────────────────────────────────────────────────────

METRICS = ["area_um2", "period_ns", "power_mw"]

# GBT hyperparameters.  Chosen to be stable on 40–50 rows:
# - n_estimators=200, max_depth=3, learning_rate=0.05: low-variance, avoids
#   overfit on ~10-fold-CV splits of ~32 rows.
# - subsample=0.8: standard stochastic GBT regularisation.
_GBT_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=2,
)

# Indicator that a model has NOT been fitted (too few rows)
_FALLBACK = "fallback"

# Obs columns that the surrogate accepts.  Order matters for feature vector.
_OBS_COLS = ["proxy_area_um2", "proxy_wns_ns", "ff_count", "cell_count", "logic_levels"]

# ── Feature engineering ────────────────────────────────────────────────────────


def _encode_config(x: dict) -> list[float]:
    """Return the 6-element config feature vector for a single config dict.

    Accepts either the optimizer-facing keys (mac_lanes / accumulator_width /
    clock_period_ns / abc_recipe / platform) or the runner-facing keys
    (lanes / acc_w / clk_ns / abc / platform).  Both styles are used in the
    corpus so we normalise here.
    """
    lanes = int(x.get("mac_lanes") or x.get("lanes") or 4)
    acc_w = int(x.get("accumulator_width") or x.get("acc_w") or 24)
    clk   = float(x.get("clock_period_ns") or x.get("clk_ns") or 5.0)

    # abc_recipe: 'area' → 1; anything else (None/'speed'/absent) → 0
    abc_raw = x.get("abc_recipe") or x.get("abc") or ""
    abc_flag = 1.0 if str(abc_raw).lower() == "area" else 0.0

    # platform: 'asap7' → 1; everything else (nangate45/sky130…) → 0
    plat_raw = x.get("platform") or "nangate45"
    plat_flag = 1.0 if str(plat_raw).lower() == "asap7" else 0.0

    # rtl_hash: treated as a categorical context feature (integer-encoded later
    # by the Surrogate which holds the hash→int mapping built during fit)
    # We return a placeholder 0.0 here; _build_feature_row replaces it.
    return [math.log2(max(lanes, 1)), float(acc_w), clk, abc_flag, plat_flag, 0.0]


def _encode_obs(obs: dict | None, means: dict) -> list[float]:
    """Return the 2×|_OBS_COLS| obs feature vector (value + present indicator).

    Missing values are replaced with the training-set column mean (from `means`)
    and their indicator is set to 0.  Present values get indicator 1.
    """
    feats: list[float] = []
    for col in _OBS_COLS:
        if obs and col in obs and obs[col] is not None:
            feats.append(float(obs[col]))
            feats.append(1.0)
        else:
            feats.append(float(means.get(col, 0.0)))
            feats.append(0.0)
    return feats


# ── Row normalisation ──────────────────────────────────────────────────────────


def _flatten_row(row: dict) -> dict:
    """Accept both flat and nested row formats and return a flat dict.

    Supported shapes:
      - results_physical.jsonl flat format: {config:{mac_lanes,...}, metrics:{...}}
      - runner flat format: {lanes:4, acc_w:24, clk_ns:5, area_um2:..., ...}
      - funnel format: {config:{...}, fidelity:..., obs:{...}, metrics:{...}}
    """
    flat: dict = {}

    # nested → flatten config
    if "config" in row and isinstance(row["config"], dict):
        flat.update(row["config"])
    # nested → flatten metrics
    if "metrics" in row and isinstance(row["metrics"], dict):
        flat.update(row["metrics"])
    # nested → flatten obs
    if "obs" in row and isinstance(row["obs"], dict):
        flat.update(row["obs"])
    # Merge remaining top-level keys (flat format)
    for k, v in row.items():
        if k not in ("config", "metrics", "obs") and k not in flat:
            flat[k] = v

    # Normalise key aliases so downstream code sees one canonical set
    for src, dst in [
        ("mac_lanes", "lanes"),
        ("accumulator_width", "acc_w"),
        ("clock_period_ns", "clk_ns"),
        ("abc_recipe", "abc"),
    ]:
        if src in flat and dst not in flat:
            flat[dst] = flat[src]

    # period_ns comes from period_min_ns in ORFS records
    if "period_ns" not in flat and "period_min_ns" in flat:
        flat["period_ns"] = flat["period_min_ns"]
    # fmax → period_ns if still missing
    if "period_ns" not in flat and "fmax_mhz" in flat and flat["fmax_mhz"]:
        flat["period_ns"] = 1000.0 / float(flat["fmax_mhz"])

    return flat


def _is_f3_row(flat: dict) -> bool:
    """Return True if this row carries at least the F3 area metric."""
    return flat.get("area_um2") is not None and flat.get("status", "ok") in (
        "ok", "mock"
    )


# ── Surrogate class ────────────────────────────────────────────────────────────


class Surrogate:
    """Multi-fidelity surrogate predicting F3 physical metrics from (x, obs)."""

    METRICS = METRICS

    def __init__(self, seed: int = 0):
        self.seed = seed
        # Per-metric: either {q: GBT} or _FALLBACK
        self._models: dict[str, Any] = {m: _FALLBACK for m in METRICS}
        # Obs column means for missing-value imputation
        self._obs_means: dict[str, float] = {c: 0.0 for c in _OBS_COLS}
        # RTL hash → integer encoding for the context feature
        self._hash_map: dict[str, int] = {}
        # Per-metric target statistics for the fallback regime
        self._target_stats: dict[str, tuple[float, float]] = {
            m: (0.0, 1.0) for m in METRICS
        }
        # Whether a real fit has been done
        self._fitted: bool = False
        self._n_rows: int = 0
        # Metadata (stored in the joblib file alongside the model)
        self.meta: dict = {}

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, rows: list[dict]) -> dict:
        """Fit the surrogate on F3 rows.

        Parameters
        ----------
        rows : list[dict]
            Each element may be flat, nested (results_physical.jsonl style),
            or funnel style.  Only rows with F3-level area_um2 are used as
            training targets; rows without area_um2 but with F2 obs are joined
            (same config key) to enrich the feature vector with proxy data.

        Returns
        -------
        dict
            Fit diagnostics: {"n_f3": int, "n_obs": int,
                              "cv_rho_<metric>": float, "cv_n_<metric>": int,
                              "fallback": bool}
        """
        flat_rows = [_flatten_row(r) for r in rows]

        # Split into F3 rows (have area_um2) and F2-only rows (have obs but no area)
        f3_rows = [r for r in flat_rows if _is_f3_row(r)]
        obs_index: dict[tuple, dict] = {}  # (lanes, acc_w, clk_ns) → obs dict
        for r in flat_rows:
            if not _is_f3_row(r):
                key = (
                    int(r.get("lanes", 0) or 0),
                    int(r.get("acc_w", 0) or 0),
                    float(r.get("clk_ns", 0.0) or 0.0),
                )
                if any(r.get(c) is not None for c in _OBS_COLS):
                    obs_index[key] = r

        n_f3 = len(f3_rows)
        self._n_rows = n_f3
        diag: dict = {"n_f3": n_f3, "n_obs": len(obs_index), "fallback": False}

        if n_f3 == 0:
            diag["fallback"] = True
            return diag

        # Build RTL hash encoding (context feature)
        all_hashes = list({str(r.get("rtl_hash", "unknown")) for r in f3_rows})
        self._hash_map = {h: i for i, h in enumerate(sorted(all_hashes))}

        # Compute obs means for imputation (from F3 rows that carry obs columns)
        obs_cols_present = {c: [] for c in _OBS_COLS}
        for r in f3_rows:
            for c in _OBS_COLS:
                if r.get(c) is not None:
                    obs_cols_present[c].append(float(r[c]))
        for c in _OBS_COLS:
            vals = obs_cols_present[c]
            if vals:
                self._obs_means[c] = float(np.mean(vals))

        # Build feature matrix X and target vectors Y
        X_list, targets = [], {m: [] for m in METRICS}
        for r in f3_rows:
            fv = self._build_feature_row(r, obs_index)
            X_list.append(fv)
            for m in METRICS:
                targets[m].append(r.get(m))

        X = np.array(X_list, dtype=float)

        # Per-metric: compute target stats then fit quantile models
        for m in METRICS:
            y_raw = targets[m]
            y_valid_idx = [i for i, v in enumerate(y_raw) if v is not None]
            y = np.array([float(y_raw[i]) for i in y_valid_idx])
            X_m = X[y_valid_idx]
            n = len(y)
            diag[f"cv_n_{m}"] = n

            if n == 0:
                self._target_stats[m] = (0.0, 1.0)
                diag[f"cv_rho_{m}"] = float("nan")
                continue

            mu_y, std_y = float(np.mean(y)), float(np.std(y))
            self._target_stats[m] = (mu_y, max(std_y, 1e-6))

            if n < 10:
                # Too few rows: fall back to mean prediction
                self._models[m] = _FALLBACK
                diag[f"cv_rho_{m}"] = float("nan")
                continue

            # Fit three quantile-GBT models
            self._models[m] = {}
            for q in (0.16, 0.50, 0.84):
                gbt = GradientBoostingRegressor(
                    loss="quantile", alpha=q,
                    random_state=self.seed,
                    **_GBT_PARAMS,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gbt.fit(X_m, y)
                self._models[m][q] = gbt

            # 5-fold CV for Spearman rho
            rho = self._cv_spearman(X_m, y, m, n_splits=5)
            diag[f"cv_rho_{m}"] = round(rho, 4)

        self._fitted = True
        self.meta.update({"n_f3": n_f3, "n_obs": len(obs_index),
                          "seed": self.seed})
        return diag

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        x: dict,
        obs: dict | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Predict (mu, sigma) for each metric given config x and optional F2 obs.

        Parameters
        ----------
        x : dict
            Config with keys: mac_lanes / lanes, accumulator_width / acc_w,
            clock_period_ns / clk_ns, abc_recipe / abc (opt), platform (opt),
            rtl_hash (opt).
        obs : dict | None
            Optional F2 observables (any subset of proxy_area_um2, proxy_wns_ns,
            ff_count, cell_count, logic_levels).

        Returns
        -------
        dict[str, tuple[float, float]]
            {metric: (mu, sigma)} for each metric in METRICS.
            If not fitted (too few rows), returns (training_mean, 2*std).
        """
        result = {}
        for m in METRICS:
            if self._models[m] is _FALLBACK or not self._fitted:
                mu, std = self._target_stats[m]
                result[m] = (float(mu), float(2.0 * std))
                continue

            fv = self._build_feature_row(x, obs_dict=obs)
            X_pred = np.array([fv], dtype=float)

            q16 = float(self._models[m][0.16].predict(X_pred)[0])
            q50 = float(self._models[m][0.50].predict(X_pred)[0])
            q84 = float(self._models[m][0.84].predict(X_pred)[0])

            mu = q50
            # sigma ≈ half-IQR of the 68% interval (like one-sigma for Gaussian)
            sigma = max((q84 - q16) / 2.0, 1e-6)
            result[m] = (mu, sigma)

        return result

    def predict_reward_stats(
        self,
        x: dict,
        obs: dict | None = None,
    ) -> tuple[float, float]:
        """Return (mu, sigma) of the final composite reward for config x.

        The reward formula follows physical_reward.compute_physical_reward:
          reward ≈ w_spd * norm_speedup + w_area * (area/AREA_REF) + ...
        We propagate uncertainties in a first-order (independent-errors)
        approximation — this is deliberately crude; use it as a UCB signal,
        not a calibrated interval.

        Weights and references mirror physical_reward.py defaults.
        """
        preds = self.predict(x, obs)

        lanes = int(x.get("mac_lanes") or x.get("lanes") or 4)
        acc_w = int(x.get("accumulator_width") or x.get("acc_w") or 24)

        # Accuracy term (analytic — no uncertainty)
        accuracy = 0.0 if acc_w <= 16 else 1.0  # rough: A16 may overflow
        w_acc = 2.0
        correctness = -50.0 * (1.0 - accuracy)

        # Speedup from period_ns prediction
        from common.constants import SW_BASELINE_CYCLES, behavioral_cycles
        SW_BASELINE_NS = SW_BASELINE_CYCLES * 10.0  # 100 MHz baseline
        cyc = behavioral_cycles(lanes)
        mu_period, sig_period = preds["period_ns"]
        # Clamp period to (0, inf) — negative predictions are unphysical
        mu_period = max(mu_period, 0.5)
        mu_latency_ns = cyc * mu_period
        mu_speedup = SW_BASELINE_NS / max(mu_latency_ns, 1.0)
        max_spd = 576.0
        norm_spd = math.log2(max(mu_speedup, 1e-3)) / math.log2(max_spd) if max_spd > 1 else 0.0
        norm_spd = max(-1.0, min(1.0, norm_spd))
        # sigma propagation: ∂(1/period)/∂period = -1/period²
        sig_speedup = (SW_BASELINE_NS / mu_latency_ns**2) * cyc * sig_period
        sig_norm_spd = (sig_speedup / (mu_speedup * math.log(max_spd))) if mu_speedup > 0 else 0.1

        AREA_REF = 19_738.0
        POWER_REF = 1_020.0
        mu_area, sig_area = preds["area_um2"]
        mu_power, sig_power = preds["power_mw"]

        mu_reward = (
            w_acc * accuracy
            + 3.0 * norm_spd
            + (-0.4) * (mu_area / AREA_REF)
            + (-0.4) * (mu_power / POWER_REF)
            + correctness
        )
        # First-order independent-errors sigma
        sig_reward = math.sqrt(
            (3.0 * sig_norm_spd) ** 2
            + (0.4 / AREA_REF * sig_area) ** 2
            + (0.4 / POWER_REF * sig_power) ** 2
        )
        return (float(mu_reward), float(sig_reward))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path) -> None:
        """Save the fitted surrogate to a joblib file."""
        payload = {
            "models": self._models,
            "obs_means": self._obs_means,
            "hash_map": self._hash_map,
            "target_stats": self._target_stats,
            "fitted": self._fitted,
            "n_rows": self._n_rows,
            "meta": self.meta,
            "seed": self.seed,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path) -> "Surrogate":
        """Load a previously saved surrogate from a joblib file."""
        payload = joblib.load(path)
        s = cls(seed=payload.get("seed", 0))
        s._models = payload["models"]
        s._obs_means = payload["obs_means"]
        s._hash_map = payload["hash_map"]
        s._target_stats = payload["target_stats"]
        s._fitted = payload["fitted"]
        s._n_rows = payload["n_rows"]
        s.meta = payload.get("meta", {})
        return s

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_feature_row(
        self,
        row_or_x: dict,
        obs_index: dict | None = None,
        obs_dict: dict | None = None,
    ) -> list[float]:
        """Build the full feature vector for one row or (x, obs) pair.

        When called with a training row, `obs_index` is the {config_key: obs}
        lookup built during fit(); when called at prediction time, `obs_dict`
        is the optional F2 observable dict passed by the caller.
        """
        flat = row_or_x  # already flat at this point

        # Config features
        lanes = int(flat.get("mac_lanes") or flat.get("lanes") or 4)
        acc_w = int(flat.get("accumulator_width") or flat.get("acc_w") or 24)
        clk   = float(flat.get("clock_period_ns") or flat.get("clk_ns") or 5.0)
        abc_raw = flat.get("abc_recipe") or flat.get("abc") or ""
        abc_flag = 1.0 if str(abc_raw).lower() == "area" else 0.0
        plat_raw = flat.get("platform") or "nangate45"
        plat_flag = 1.0 if str(plat_raw).lower() == "asap7" else 0.0
        h = str(flat.get("rtl_hash", "unknown"))
        hash_cat = float(self._hash_map.get(h, len(self._hash_map)))  # unseen = n+1

        cfg_feats = [math.log2(max(lanes, 1)), float(acc_w), clk, abc_flag, plat_flag, hash_cat]

        # Resolve obs: priority = explicit obs_dict > obs from the row itself >
        # obs looked up from obs_index (training time join)
        obs: dict | None = None
        if obs_dict is not None:
            obs = obs_dict
        elif obs_index is not None:
            key = (lanes, acc_w, clk)
            obs = obs_index.get(key)
        # Also pick up obs columns that are present directly in the row
        row_obs: dict = {}
        for c in _OBS_COLS:
            if flat.get(c) is not None:
                row_obs[c] = flat[c]
        if row_obs:
            if obs is None:
                obs = row_obs
            else:
                obs = {**row_obs, **obs}  # row_obs provides defaults, obs overrides

        obs_feats = _encode_obs(obs, self._obs_means)
        return cfg_feats + obs_feats

    def _cv_spearman(
        self, X: np.ndarray, y: np.ndarray, metric: str, n_splits: int = 5
    ) -> float:
        """Compute mean Spearman rho via stratified K-fold cross-validation.

        Uses only q=0.50 (median) predictions for the rank correlation.
        Falls back to leave-one-out if n < 2*n_splits.
        """
        from scipy.stats import spearmanr
        n = len(y)
        if n < 2 * n_splits:
            n_splits = max(2, n // 2)

        indices = np.arange(n)
        fold_size = n // n_splits
        rhos = []
        for i in range(n_splits):
            val_idx = indices[i * fold_size:(i + 1) * fold_size]
            if len(val_idx) == 0:
                continue
            train_idx = np.concatenate([indices[:i * fold_size],
                                        indices[(i + 1) * fold_size:]])
            if len(train_idx) < 3:
                continue
            gbt = GradientBoostingRegressor(
                loss="quantile", alpha=0.5,
                random_state=self.seed,
                **_GBT_PARAMS,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gbt.fit(X[train_idx], y[train_idx])
            y_pred = gbt.predict(X[val_idx])
            if len(y_pred) < 2:
                continue
            rho, _ = spearmanr(y[val_idx], y_pred)
            if not math.isnan(rho):
                rhos.append(rho)

        return float(np.mean(rhos)) if rhos else float("nan")


# ── Self-test / data mining entrypoint ────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Run the fit script if invoked directly
    script = Path(__file__).parent / "fit_surrogate.py"
    if script.exists():
        exec(compile(script.read_text(), str(script), "exec"))
    else:
        print("fit_surrogate.py not found — run it directly for data mining + CP3 validation")
        sys.exit(1)

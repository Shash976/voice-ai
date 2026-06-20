#!/usr/bin/env python3
"""campaign_data.py — load funnel-optimizer campaign logs into tidy structures.

A campaign JSONL (written by run_funnel_optimizer.py) has one episode per line:

    {"ts", "campaign_id", "episode", "config": {...}, "actions": [...],
     "fidelity": "F0|F1|F2|F3", "step_rewards": [...], "episode_reward": float,
     "f3_reward": float|null, "best_reward": float|null, "spent_s": float,
     "episode_s": float}

This module flattens those rows (config keys promoted to top-level ``p_<name>``
columns), infers an Optuna distribution per parameter from the observed values,
and can reconstruct an Optuna study so both the static report and the live
dashboard share one source of truth.

Objective convention (maximize): an episode's value is its terminal ``f3_reward``
when it reached F3, else its ``episode_reward`` (the accumulated failure-ladder
penalty for killed/aborted episodes). Killed episodes are kept — seeing where in
parameter space designs die is part of "how reward changed with parameters".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ── default location ──────────────────────────────────────────────────────────
def _find_latest_campaign_log() -> Path:
    """Return the most-recently-modified results_funnel_campaigns.jsonl under
    optimizer/campaigns/<design>/<platform>/; fall back to the old flat path."""
    campaigns_root = Path(__file__).resolve().parents[1] / "campaigns"
    if campaigns_root.exists():
        logs = sorted(
            campaigns_root.rglob("results_funnel_campaigns.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if logs:
            return logs[0]
    return Path(__file__).resolve().parents[1] / "results_funnel_campaigns.jsonl"


DEFAULT_LOG = _find_latest_campaign_log()

_FIDELITY_ORDER = ["F0", "F1", "F2", "F3"]


# ── row loading ───────────────────────────────────────────────────────────────

def load_campaign_rows(
    log_path: str | Path,
    campaign: str | None = None,
) -> list[dict]:
    """Read a campaign JSONL into a list of episode dicts.

    ``campaign`` selects which campaign_id to keep:
      - None or "latest": only the most-recently-started campaign in the file
      - "all": every row
      - "<campaign_id>": exact match
    Malformed lines are skipped.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"campaign log not found: {log_path}")

    rows: list[dict] = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "config" in row:
                rows.append(row)

    if not rows:
        return rows

    sel = (campaign or "latest").strip()
    if sel == "all":
        return rows

    if sel == "latest":
        # Pick campaign_id with the largest starting ts (id is campaign_<seed>_<ts>).
        def _start_ts(cid: str) -> float:
            try:
                return float(cid.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                return 0.0
        cids = {r.get("campaign_id", "") for r in rows}
        sel = max(cids, key=_start_ts)

    return [r for r in rows if r.get("campaign_id") == sel]


def episode_value(row: dict) -> float | None:
    """Maximize-objective for one episode: f3_reward if present, else episode_reward."""
    v = row.get("f3_reward")
    if v is None:
        v = row.get("episode_reward")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── param-domain inference ────────────────────────────────────────────────────

@dataclass
class ParamSpec:
    name: str
    kind: str           # "int" | "float" | "cat"
    low: float = 0.0
    high: float = 1.0
    choices: list = field(default_factory=list)


def infer_param_specs(rows: Iterable[dict]) -> dict[str, ParamSpec]:
    """Infer one ParamSpec per config key seen across all rows.

    A param is categorical if any observed value is a non-bool string; integer if
    every observed value is an int (or float with no fractional part); float
    otherwise. Bounds are the observed min/max (slightly padded so endpoints are
    interior, which keeps Optuna distributions happy).
    """
    vals: dict[str, list] = {}
    for row in rows:
        cfg = row.get("config") or {}
        for k, v in cfg.items():
            vals.setdefault(k, []).append(v)

    specs: dict[str, ParamSpec] = {}
    for name, observed in vals.items():
        non_null = [v for v in observed if v is not None]
        if not non_null:
            continue
        if any(isinstance(v, str) for v in non_null):
            choices = sorted({str(v) for v in non_null})
            specs[name] = ParamSpec(name, "cat", choices=choices)
            continue
        nums = [float(v) for v in non_null if isinstance(v, (int, float))]
        if not nums:
            choices = sorted({str(v) for v in non_null})
            specs[name] = ParamSpec(name, "cat", choices=choices)
            continue
        lo, hi = min(nums), max(nums)
        is_int = all(
            isinstance(v, int) or (isinstance(v, float) and float(v).is_integer())
            for v in non_null
        )
        if hi <= lo:                      # constant param → tiny interval
            hi = lo + (1 if is_int else 1e-9)
        specs[name] = ParamSpec(name, "int" if is_int else "float", low=lo, high=hi)
    return specs


# ── Optuna study reconstruction ───────────────────────────────────────────────

def build_study(
    rows: list[dict],
    *,
    study_name: str = "funnel",
    storage: Any | None = None,
    only_params: list[str] | None = None,
):
    """Reconstruct an Optuna study (direction=maximize) from campaign rows.

    Each episode with a finite value becomes a COMPLETE trial. ``fidelity`` and a
    few scalars are stored as user_attrs so the dashboard can colour/inspect them.
    Pass ``storage`` (e.g. a JournalStorage) to persist for optuna-dashboard;
    omit it for an in-memory study (used by the static report).

    ``only_params`` restricts which config keys become Optuna params (importance /
    slice plots need a consistent space — handy when mixing tiers).
    """
    import optuna
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
    from optuna.trial import TrialState, create_trial

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    specs = infer_param_specs(rows)
    if only_params is not None:
        keep = set(only_params)
        specs = {k: v for k, v in specs.items() if k in keep}

    def _dist(spec: ParamSpec):
        if spec.kind == "cat":
            return CategoricalDistribution(choices=spec.choices)
        if spec.kind == "int":
            return IntDistribution(low=int(spec.low), high=int(spec.high))
        return FloatDistribution(low=spec.low, high=spec.high)

    dists = {name: _dist(spec) for name, spec in specs.items()}

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    # Skip rows already present (resume / live append).
    start = len(study.get_trials(deepcopy=False))

    added = 0
    for row in rows[start:]:
        value = episode_value(row)
        if value is None:
            continue
        cfg = row.get("config") or {}
        params: dict[str, Any] = {}
        tdists: dict[str, Any] = {}
        for name, spec in specs.items():
            if name not in cfg or cfg[name] is None:
                continue
            raw = cfg[name]
            if spec.kind == "cat":
                params[name] = str(raw)
            elif spec.kind == "int":
                params[name] = int(round(float(raw)))
            else:
                params[name] = float(raw)
            tdists[name] = dists[name]
        if not params:
            continue
        attrs = {
            "fidelity": row.get("fidelity"),
            "episode": row.get("episode"),
            "campaign_id": row.get("campaign_id"),
            "spent_s": row.get("spent_s"),
            "reached_f3": row.get("f3_reward") is not None,
        }
        trial = create_trial(
            state=TrialState.COMPLETE,
            value=value,
            params=params,
            distributions=tdists,
            user_attrs=attrs,
        )
        study.add_trial(trial)
        added += 1

    return study, added, specs


# ── high-level container ──────────────────────────────────────────────────────

@dataclass
class CampaignData:
    rows: list[dict]
    specs: dict[str, ParamSpec]
    campaign_id: str

    @classmethod
    def load(cls, log_path: str | Path, campaign: str | None = None) -> "CampaignData":
        rows = load_campaign_rows(log_path, campaign)
        specs = infer_param_specs(rows)
        cid = rows[0].get("campaign_id", "?") if rows else "?"
        if campaign in (None, "latest") and rows:
            cid = rows[-1].get("campaign_id", cid)
        elif campaign == "all":
            cid = "all"
        return cls(rows=rows, specs=specs, campaign_id=cid)

    def fidelity_counts(self) -> dict[str, int]:
        counts = {f: 0 for f in _FIDELITY_ORDER}
        for r in self.rows:
            f = r.get("fidelity")
            counts[f] = counts.get(f, 0) + 1
        return counts

    def values(self) -> list[float | None]:
        return [episode_value(r) for r in self.rows]

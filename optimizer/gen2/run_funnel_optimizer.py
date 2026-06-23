#!/usr/bin/env python3
"""run_funnel_optimizer.py — live campaign driver for the gen2 funnel optimizer.

Gen2 counterpart of gen1/run_physical_optimizer.py.  Drives a loop of:
  1. CandidateGenerator.suggest() → next config
  2. FunnelEnv.reset(config)      → run F0, get initial state
  3. PromotionAgent.act(state) → step(action) → repeat until done=True
  4. CandidateGenerator.update(config, terminal_reward, fidelity)
  5. PromotionAgent.update per step (online LinUCB update)

Logs one JSONL line per episode to optimizer/campaigns/<design>/<platform>/results_funnel_campaigns.jsonl.
Prints a running incumbent line and a summary on exit.

CLI
---
  python3 optimizer/gen2/run_funnel_optimizer.py \\
      --design tinymac_accel --platform nangate45 \\
      --budget-hours 4 \\
      --max-tier 1 \\
      --sampler tpe|surrogate_ucb|random \\
      --promotion fixed|linucb|random \\
      --seed 0 \\
      --table optimizer/results/gen2/results_funnel.jsonl  # omit for live mode
      --surrogate optimizer/results/gen2/surrogate_n45.joblib   # default: auto-detect

Table mode (--table given): replays logged observations, charges recorded cost
against a simulated wall-clock budget.  PHYSICAL_MOCK=1 activates mock metrics
for live mode (no real ORFS calls).

Entry point is also available via the compat shim optimizer/run_funnel_optimizer.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── bootstrap: make optimizer/ root importable ───────────────────────────────
import pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

# Force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
except Exception:
    pass

# ── imports (all defensive with clear error messages) ─────────────────────────

try:
    from gen2.funnel import FunnelEnv, load_table
    _FUNNEL_OK = True
except Exception as _e:
    _FUNNEL_OK = False
    _FUNNEL_ERR = str(_e)

try:
    from gen2.candidates import CandidateGenerator, _fallback_space
    _CAND_OK = True
except Exception as _e:
    _CAND_OK = False
    _CAND_ERR = str(_e)

try:
    from gen2.promotion_agent import (
        FixedGateAgent,
        PromotionAgent,
        RandomPromotionAgent,
    )
    _PROMO_OK = True
except Exception as _e:
    _PROMO_OK = False
    _PROMO_ERR = str(_e)


# ── constants ─────────────────────────────────────────────────────────────────

_OPT_ROOT = Path(__file__).resolve().parents[1]
_CAMPAIGNS_ROOT = _OPT_ROOT / "campaigns"
_DEFAULT_SURROGATE = _OPT_ROOT / "results" / "gen2" / "surrogate_n45.joblib"
_DEFAULT_SPACE_YAML = Path(__file__).resolve().parent / "search_space_funnel.yaml"

# Fidelity labels in promotion order
_FIDELITY_ORDER = ["F0", "F1", "F2", "F3"]


# ── helper: load surrogate ────────────────────────────────────────────────────

def _load_surrogate(path: str | Path | None) -> Any | None:
    """Try to load a Surrogate from path; return None on failure."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        from gen2.surrogate import Surrogate
        s = Surrogate.load(p)
        return s
    except Exception as exc:   # noqa: BLE001
        print(f"  [WARNING] could not load surrogate from {p}: {exc}")
        return None


# ── helper: build space dict ──────────────────────────────────────────────────

def _build_space(
    design: str | None,
    platform: str,
    max_tier: int,
    space_yaml: "str | Path | None" = None,
) -> dict:
    """Return a space dict, trying KnobRegistry+DesignSpec first then _fallback_space.

    KnobRegistry.space() normalizes str designs via DesignSpec.load() and emits
    design.params axes under their canonical names (mac_lanes, accumulator_width for
    tinymac; empty for designs like gcd that have no RTL params).

    Fixed knobs: axes listed under ``fixed:`` in space_yaml are removed from the
    returned space so the CandidateGenerator never samples them.  This preserves
    the historical 4-axis tinymac space for ``--max-tier 1`` (CORE_UTILIZATION is
    fixed at 40 per search_space_funnel.yaml).

    Fallback: if KnobRegistry/DesignSpec are unavailable, use _fallback_space()
    (hardcoded 4-axis tinymac space).
    """
    # Read fixed knobs from the space YAML (design-specific overrides)
    fixed_knob_names: set = set()
    if space_yaml is not None:
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            _p = _Path(space_yaml)
            if _p.exists():
                with open(_p, encoding="utf-8") as _f:
                    _raw = _yaml.safe_load(_f)
                fixed_knob_names = set((_raw.get("fixed") or {}).keys())
        except Exception:
            pass

    try:
        from common.knobs import KnobRegistry

        reg = KnobRegistry.load()
        # reg.space() accepts str and normalizes via DesignSpec.load() internally.
        sp = reg.space(max_tier=max_tier, design=design, platform=platform)
        if sp:
            # Remove axes that are fixed constants in the space YAML
            if fixed_knob_names:
                sp = {k: v for k, v in sp.items() if k not in fixed_knob_names}
            return sp
    except (ImportError, AttributeError, Exception):
        pass

    # Fallback: hardcoded 4-axis tinymac space (only appropriate for tinymac)
    return _fallback_space()


# ── helper: build promotion agent ─────────────────────────────────────────────

def _make_promotion_agent(name: str, seed: int) -> Any:
    """Construct promotion agent by name."""
    if not _PROMO_OK:
        raise RuntimeError(f"promotion_agent.py not available: {_PROMO_ERR}")
    if name == "fixed":
        return FixedGateAgent(seed=seed)
    elif name == "linucb":
        return PromotionAgent(seed=seed)
    elif name == "random":
        return RandomPromotionAgent(seed=seed)
    else:
        raise ValueError(f"Unknown promotion agent: {name!r}; use fixed|linucb|random")


# ── helper: extract F3 reward from episode info ────────────────────────────────

def _extract_f3_reward(
    episode_reward: float,
    fidelity_reached: str,
    episode_done: bool,
) -> float | None:
    """Return the terminal F3 reward, or None if the episode did not reach F3."""
    if episode_done and fidelity_reached == "F3":
        return float(episode_reward)
    return None


# ── campaign loop ─────────────────────────────────────────────────────────────

def run_campaign(
    *,
    design: str | None,
    platform: str,
    budget_s: float,
    max_tier: int,
    sampler: str,
    promotion: str,
    seed: int,
    table_path: str | Path | None,
    surrogate_path: str | Path | None,
    results_path: str | Path,
    space_yaml: str | Path,
    verbose: bool = True,
) -> dict:
    """Run one funnel optimizer campaign.

    Returns a summary dict: {best_config, best_reward, n_episodes, elapsed_s, ...}
    """
    if not _FUNNEL_OK:
        raise RuntimeError(f"FunnelEnv not available: {_FUNNEL_ERR}")
    if not _CAND_OK:
        raise RuntimeError(f"CandidateGenerator not available: {_CAND_ERR}")

    t0 = time.time()
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_id = f"campaign_{seed}_{int(t0)}"

    # ── load surrogate ─────────────────────────────────────────────────────────
    # Auto-detect default surrogate if not specified
    if surrogate_path is None and _DEFAULT_SURROGATE.exists():
        surrogate_path = _DEFAULT_SURROGATE
    surrogate = _load_surrogate(surrogate_path)
    if verbose and surrogate is not None:
        print(f"  Surrogate loaded from {surrogate_path} "
              f"(fitted={surrogate._fitted}, n_rows={surrogate._n_rows})")
    elif verbose:
        print(f"  No surrogate (UCB scoring disabled)")

    # ── build space ────────────────────────────────────────────────────────────
    space = _build_space(design, platform, max_tier, space_yaml=space_yaml)
    if verbose:
        print(f"  Space: {len(space)} axes: {list(space.keys())}")

    # ── load table (if given) ──────────────────────────────────────────────────
    table = None
    if table_path is not None:
        tp = Path(table_path)
        if tp.exists():
            table = load_table(tp)
            if verbose:
                f3_count = sum(1 for e in table.values() if "F3" in e)
                print(f"  Table loaded: {len(table)} configs, {f3_count} with F3 rows")
                if f3_count == 0:
                    print(f"  [INFO] Table has no F3 rows: commit actions will return "
                          f"the failure-ladder penalty (-20) per documented honesty rule.")
        else:
            print(f"  [WARNING] Table not found at {tp}; running in live mode.")

    # ── build FunnelEnv ────────────────────────────────────────────────────────
    env = FunnelEnv(
        space_yaml=space_yaml,
        platform=platform,
        budget_s=budget_s,
        surrogate=surrogate,
        table=table,
        results_path=results_path.parent / f"funnel_{campaign_id}.jsonl",
        seed=seed,
        design=design,
        max_tier=max_tier,
        active_space=space,  # pass the KnobRegistry space so validation uses its bounds
    )

    # ── build agents ───────────────────────────────────────────────────────────
    gen = CandidateGenerator(
        space=space,
        sampler=sampler,
        surrogate=surrogate,
        seed=seed,
        kappa=1.0,
        grid_snap=(table is not None),   # snap to table grid in table mode
    )
    promo = _make_promotion_agent(promotion, seed=seed)

    # ── campaign tracking ──────────────────────────────────────────────────────
    best_reward = float("-inf")
    best_config: dict | None = None
    n_episodes = 0
    n_killed = 0
    n_f3 = 0
    per_fidelity_counts: dict[str, int] = {f: 0 for f in _FIDELITY_ORDER}

    if verbose:
        print(f"\n  Campaign {campaign_id}")
        print(f"  sampler={sampler} promotion={promotion} budget={budget_s/3600:.2f}h "
              f"seed={seed} table={'yes' if table else 'no'}")
        print(f"  {'Episode':>8} {'Fidelity':>8} {'Reward':>9} {'Best':>9} "
              f"{'Spent(h)':>9} {'Config'}")
        print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*40}")

    log_rows: list[dict] = []

    while env.spent_s < budget_s:
        # ── generate candidate ─────────────────────────────────────────────────
        try:
            config = gen.suggest()
        except Exception as exc:   # noqa: BLE001
            print(f"  [ERROR] CandidateGenerator.suggest() failed: {exc}")
            break

        # ── run episode ────────────────────────────────────────────────────────
        try:
            state = env.reset(config)
        except (ValueError, KeyError) as exc:
            # Invalid config (not in table, or constraint violation)
            gen.update(config, reward=-100.0, fidelity="invalid")
            continue
        except Exception as exc:   # noqa: BLE001
            gen.update(config, reward=-100.0, fidelity="invalid")
            print(f"  [WARN] env.reset failed: {exc}")
            continue

        episode_reward_acc = 0.0
        episode_done = False
        fidelity_reached = "F0"
        episode_actions: list[str] = []
        episode_step_rewards: list[float] = []
        episode_t0 = time.time()

        while not episode_done:
            if env.spent_s + env._episode_spent_s >= budget_s:
                # Over budget mid-episode
                break

            action = promo.act(state)
            episode_actions.append(action)

            try:
                next_state, reward, episode_done, info = env.step(action)
                print(f"    └── [STEP] Gate: {fidelity_reached} -> Action Chosen: {action.upper()} -> Step Reward: {reward:+.3f}")
            except Exception as exc:   # noqa: BLE001
                print(f"  [WARN] env.step({action!r}) failed: {exc}")
                episode_done = True
                reward = 0.0
                info = {"fidelity": fidelity_reached, "action": action}

            promo.update(state, action, reward)
            state = next_state
            episode_reward_acc += reward
            episode_step_rewards.append(reward)

            fid = info.get("fidelity", fidelity_reached)
            if fid in _FIDELITY_ORDER:
                if _FIDELITY_ORDER.index(fid) > _FIDELITY_ORDER.index(fidelity_reached):
                    fidelity_reached = fid

            if episode_done:
                act = info.get("action", action)
                if act == "kill":
                    n_killed += 1

        # ── episode complete ───────────────────────────────────────────────────
        n_episodes += 1
        per_fidelity_counts[fidelity_reached] = (
            per_fidelity_counts.get(fidelity_reached, 0) + 1
        )

        # F3 terminal reward
        f3_reward: float | None = _extract_f3_reward(
            episode_reward_acc, fidelity_reached, episode_done
        )
        if f3_reward is not None:
            gen.update(config, f3_reward, fidelity="F3")
            n_f3 += 1
            if f3_reward > best_reward:
                best_reward = f3_reward
                best_config = dict(config)
        else:
            gen.update(config, episode_reward_acc, fidelity=fidelity_reached)

        # Update incumbent in env (for state slot [16])
        # env tracks its own incumbent; we track ours separately for logging

        # ── logging ────────────────────────────────────────────────────────────
        spent_h = env.spent_s / 3600.0
        log_row = {
            "ts":            time.time(),
            "campaign_id":   campaign_id,
            "episode":       n_episodes,
            "config":        config,
            "actions":       episode_actions,
            "fidelity":      fidelity_reached,
            "step_rewards":  episode_step_rewards,
            "episode_reward": episode_reward_acc,
            "f3_reward":     f3_reward,
            "best_reward":   best_reward if best_reward != float("-inf") else None,
            "spent_s":       round(env.spent_s, 2),
            "episode_s":     round(time.time() - episode_t0, 3),
        }
        log_rows.append(log_row)

        try:
            with open(results_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(log_row) + "\n")
        except OSError:
            pass

        if verbose:
            cfg_items = []
            for axis_name in space.keys():
                val = config.get(axis_name, '?')
                # If it's a float, format it cleanly so it doesn't clutter the screen
                if isinstance(val, float):
                    cfg_items.append(f"{axis_name}={val:.3f}")
                else:
                    cfg_items.append(f"{axis_name}={val}")
            cfg_str = " | ".join(cfg_items)
            r_str = f"{episode_reward_acc:+.3f}" if f3_reward is None else f"{f3_reward:+.3f}(F3)"
            best_str = f"{best_reward:+.3f}" if best_reward != float("-inf") else "     —"
            print(f"  {n_episodes:>8d} {fidelity_reached:>8} {r_str:>9} "
                  f"{best_str:>9} {spent_h:>8.3f}h  {cfg_str}")

    # ── summary ────────────────────────────────────────────────────────────────
    elapsed_s = time.time() - t0
    summary = {
        "campaign_id":        campaign_id,
        "best_config":        best_config,
        "best_reward":        best_reward if best_reward != float("-inf") else None,
        "n_episodes":         n_episodes,
        "n_f3":               n_f3,
        "n_killed":           n_killed,
        "per_fidelity":       per_fidelity_counts,
        "elapsed_s":          round(elapsed_s, 1),
        "simulated_s":        round(env.spent_s, 1),
        "sampler":            sampler,
        "promotion":          promotion,
        "seed":               seed,
        "platform":           platform,
    }

    if verbose:
        print(f"\n  Summary")
        print(f"  {'Episodes':>15}: {n_episodes}")
        print(f"  {'F3 commits':>15}: {n_f3}")
        print(f"  {'Killed':>15}: {n_killed}")
        print(f"  {'Per fidelity':>15}: {per_fidelity_counts}")
        print(f"  {'Best reward':>15}: "
              f"{best_reward:.4f}" if best_reward != float("-inf") else "  (none)")
        print(f"  {'Best config':>15}: {best_config}")
        print(f"  {'Elapsed':>15}: {elapsed_s:.1f}s real, "
              f"{env.spent_s/3600:.3f}h simulated")
        print(f"  Results → {results_path}")
        per_ep = elapsed_s / max(n_episodes, 1)
        per_f3 = elapsed_s / max(n_f3, 1)
        print(f"  Throughput: {per_ep:.1f}s/episode, {per_f3:.1f}s/F3")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Gen2 funnel optimizer campaign driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--design", default="tinymac_accel",
                   help="Design identifier (for KnobRegistry; falls back to hardcoded space)")
    p.add_argument("--platform", default="nangate45",
                   help="Target platform (nangate45 or asap7)")
    p.add_argument("--budget-hours", type=float, default=4.0, dest="budget_hours",
                   help="Campaign budget in hours (real or simulated)")
    p.add_argument("--max-tier", type=int, default=1, dest="max_tier",
                   help="Maximum knob tier from KnobRegistry (1 = core axes only)")
    p.add_argument("--sampler", choices=["tpe", "surrogate_ucb", "random"],
                   default="tpe",
                   help="Candidate generator sampler")
    p.add_argument("--promotion", choices=["fixed", "linucb", "random"],
                   default="fixed",
                   help="Promotion policy agent")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")
    p.add_argument("--table", default=None, dest="table",
                   help="Path to results_funnel.jsonl (table mode); omit for live mode")
    p.add_argument("--surrogate", default=None, dest="surrogate",
                   help="Path to surrogate .joblib (default: auto-detect surrogate_n45.joblib)")
    p.add_argument("--out", default=None, dest="out",
                   help="JSONL output path for campaign summary log "
                        "(default: campaigns/<design>/<platform>/results_funnel_campaigns.jsonl)")
    p.add_argument("--space-yaml", default=str(_DEFAULT_SPACE_YAML), dest="space_yaml",
                   help="Path to search_space_funnel.yaml")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-episode output")

    args = p.parse_args()

    if not _FUNNEL_OK:
        print(f"ERROR: FunnelEnv not available: {_FUNNEL_ERR}")
        sys.exit(1)
    if not _CAND_OK:
        print(f"ERROR: CandidateGenerator not available: {_CAND_ERR}")
        sys.exit(1)

    if args.out is not None:
        out_path = Path(args.out)
    else:
        design_slug = args.design or "unknown"
        out_path = _CAMPAIGNS_ROOT / design_slug / args.platform / "results_funnel_campaigns.jsonl"

    budget_s = args.budget_hours * 3600.0
    is_mock = os.environ.get("PHYSICAL_MOCK", "").strip() in ("1", "true", "True", "yes")
    if is_mock:
        print(f"  [MOCK MODE] PHYSICAL_MOCK=1 — using mock metrics")

    run_campaign(
        design=args.design,
        platform=args.platform,
        budget_s=budget_s,
        max_tier=args.max_tier,
        sampler=args.sampler,
        promotion=args.promotion,
        seed=args.seed,
        table_path=args.table,
        surrogate_path=args.surrogate,
        results_path=out_path,
        space_yaml=Path(args.space_yaml),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

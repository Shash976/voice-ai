"""fit_surrogate.py — data mining + fit + CP3 validation for the Surrogate.

This script:
1. Mines the 46 fully-built nangate45 variants from:
     /home/shashwat/voiceAI/physical/orfs/make/{reports,logs}/nangate45/tinymac_accel/
2. Enriches each row with F2 proxy observables from:
     /home/shashwat/voiceAI/physical/orfs/make/proxy/
3. Also reads sweep_results*.csv if present.
4. Fits the Surrogate on all extracted rows.
5. Reports per-metric 5-fold CV Spearman rho vs CP3 thresholds:
     - area_um2  ≥ 0.80  (CP3 requirement)
     - period_ns ≥ 0.70  (CP3 requirement)
     - power_mw  (informational — expected worse due to activity-factor fiction)
6. Runs sanity predictions:
     - area(L1) < area(L32)          [LANES dominates area, must hold]
     - sigma larger for OOD config (asap7 platform or extreme clk) than in-dist
7. Saves the fitted model to optimizer/surrogate_n45.joblib.

NOTE: physical_runner.py is being edited concurrently.  We vendor the same
report-parsing regexes here rather than importing to avoid transient import
failures.  The regexes are intentionally identical to physical_runner._parse_metrics.

Run with:
    python3 optimizer/fit_surrogate.py
or via:
    python3 optimizer/surrogate.py  (delegates here)
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# Bootstrap: make optimizer/ root importable (gen2/ is one level below it)
import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

# ── Paths ──────────────────────────────────────────────────────────────────────

_REPO     = Path(__file__).resolve().parent.parent.parent
_OPTIMIZER = Path(__file__).resolve().parent.parent       # optimizer/
_RUN_DIR  = _REPO / "physical" / "orfs" / "runs"
_RPT_DIR  = _RUN_DIR / "reports" / "nangate45" / "tinymac_accel"
_LOG_DIR  = _RUN_DIR / "logs"    / "nangate45" / "tinymac_accel"
_PROXY_DIR = _RUN_DIR / "proxy"
# Legacy classic-flow location for sweep CSVs (kept for back-compat; usually empty).
_MAKE_DIR = _REPO / "physical" / "orfs" / "make"
# Where live campaigns now log per-fidelity rows: campaigns/<design>/<platform>/*.jsonl
_CAMPAIGN_DIR = _OPTIMIZER / "campaigns"
_OUT_MODEL = Path(__file__).resolve().parent.parent / "results" / "gen2" / "surrogate_n45.joblib"

# ── Vendored regexes from physical_runner.py (identical; keep in sync) ────────
# Source: physical_runner._parse_metrics, checked 2026-06-11

_RE_AREA     = re.compile(r"Design area\s+([\d.]+)\s+um\^2\s+([\d.]+)%")
_RE_PERIOD   = re.compile(r"period_min\s*=\s*([\d.]+).*?fmax\s*=\s*([\d.]+)")
_RE_VIOLS    = re.compile(r"setup violation count\s+(\d+)")


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def _last_float_on_line(text: str, prefix: str) -> float | None:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", line)
            if nums:
                return float(nums[-1])
    return None


# ── Variant name parser ────────────────────────────────────────────────────────

def _parse_variant_name(vname: str) -> dict | None:
    """Parse L{lanes}_A{acc_w}_c{clk}[_u{util}][_d{density}][_area|_speed]."""
    m = re.match(
        r"L(\d+)_A(\d+)_c(\d+p?\d*)"
        r"(?:_u(\d+))?"
        r"(?:_d(\d+p\d+))?"
        r"(?:_(area|speed))?$",
        vname,
    )
    if not m:
        return None
    lanes = int(m.group(1))
    acc_w = int(m.group(2))
    clk_ns = float(m.group(3).replace("p", "."))
    util = int(m.group(4)) if m.group(4) else 40
    density = float(m.group(5).replace("p", ".")) if m.group(5) else 0.60
    abc = m.group(6)  # 'area', 'speed', or None
    return {
        "lanes": lanes, "acc_w": acc_w, "clk_ns": clk_ns,
        "util": util, "density": density, "abc": abc,
    }


# ── F3 metric extraction ───────────────────────────────────────────────────────

def _extract_f3_metrics(variant: str) -> dict | None:
    """Parse the 6_finish.rpt + 6_report.log + 6_report.json for one variant.

    Returns a dict with F3 targets or None if the report is missing/unparseable.
    """
    rpt  = _RPT_DIR / variant / "6_finish.rpt"
    rlog = _LOG_DIR / variant / "6_report.log"
    rjson = _LOG_DIR / variant / "6_report.json"

    rpt_txt  = _read(rpt)
    rlog_txt = _read(rlog)

    if not rpt_txt and not rlog_txt:
        return None

    out: dict = {}

    # Area from 6_report.log
    m = _RE_AREA.search(rlog_txt)
    if m:
        out["area_um2"] = float(m.group(1))
        out["util_pct"] = float(m.group(2))

    # Timing from 6_finish.rpt
    m = _RE_PERIOD.search(rpt_txt)
    if m:
        out["period_ns"]  = float(m.group(1))   # period_min_ns
        out["fmax_mhz"]   = float(m.group(2))

    raw_wns = _last_float_on_line(rpt_txt, "wns")
    raw_tns = _last_float_on_line(rpt_txt, "tns")
    if raw_wns is not None:
        out["wns_ns"] = raw_wns
    if raw_tns is not None:
        out["tns_ns"] = raw_tns

    m = _RE_VIOLS.search(rpt_txt)
    if m:
        out["setup_viol"] = int(m.group(1))

    # Power from finish.rpt "Total" line (Watts → mW)
    for line in rpt_txt.splitlines():
        if line.strip().startswith("Total"):
            nums = re.findall(r"\d+\.?\d*(?:[eE][-+]?\d+)?", line)
            if len(nums) >= 4:
                out["power_mw"] = float(nums[3]) * 1000.0
            break

    # timing_met
    if "setup_viol" in out:
        out["timing_met"] = out["setup_viol"] == 0
    elif "wns_ns" in out:
        out["timing_met"] = out["wns_ns"] >= 0.0

    # Structured counts from 6_report.json (richer than regex)
    if rjson.exists():
        try:
            jd = json.loads(rjson.read_text())
            # FF count = sequential_cell instance count
            ff = jd.get("finish__design__instance__count__class:sequential_cell")
            if ff is not None:
                out["ff_count"] = int(ff)
            # total standard-cell count
            sc = jd.get("finish__design__instance__count__stdcell")
            if sc is not None:
                out["cell_count"] = int(sc)
        except (json.JSONDecodeError, OSError):
            pass

    # Need at least area and timing for a usable row
    if "area_um2" not in out:
        return None

    out["status"] = "ok"
    return out


# ── F2 proxy metric extraction ─────────────────────────────────────────────────

def _extract_proxy_obs(proxy_key: str) -> dict:
    """Parse synth.log and sta.log from the proxy dir for F2 observables.

    proxy_key: the directory name under /proxy/, e.g. 'L4_A24_c2p0'.
    """
    pdir = _PROXY_DIR / proxy_key
    if not pdir.is_dir():
        return {}

    synth_log = _read(pdir / "synth.log")
    sta_log   = _read(pdir / "sta.log")

    obs: dict = {}

    # Proxy area: "Chip area for module '\tinymac_accel': <float>"
    m = re.search(r"Chip area for module.*?:\s*([\d.]+)", synth_log)
    if m:
        # proxy cell area (no inflation); stored as proxy_area_um2 (raw cell area)
        obs["proxy_area_um2"] = float(m.group(1))

    # Proxy WNS from STA
    wns = _last_float_on_line(sta_log, "wns")
    if wns is not None:
        obs["proxy_wns_ns"] = wns

    return obs


def _proxy_key_for_variant(variant: str) -> str:
    """Map a full variant name to the canonical proxy key (L{n}_A{w}_c{clk}).

    The proxy dirs don't have util/density/abc suffixes.
    """
    m = re.match(r"(L\d+_A\d+_c[\dp]+)", variant)
    return m.group(1) if m else variant


# ── Campaign JSONL mining (the real F3 data lives here now) ─────────────────────
# The historical report tree (physical/orfs/runs/reports/...) is no longer
# present; live campaigns write per-fidelity rows to
# campaigns/<design>/<platform>/*.jsonl with the full PPA dict under "obs".

def _rtl_hash_from_gds(gds: str | None) -> str:
    """Extract the 6–8 hex RTL/knob content hash from a GDS path variant name.

    Variant names look like 'gcd_L4_A24_c1p233_r5009f224'; the '_r<hex>' tail is
    the content hash physical_runner embeds.  Used as the surrogate's context
    feature so distinct designs/RTL versions do not alias.
    """
    if not gds:
        return "unknown"
    m = re.search(r"_r([0-9a-fA-F]{6,8})", str(gds))
    return m.group(1) if m else "unknown"


def _campaign_row_to_training(row: dict) -> dict | None:
    """Convert one campaign JSONL F3 row into a flat training row, or None.

    Only F3 rows with status ok/mock and a parsed area_um2 are usable as
    training targets.  Returns the same flat schema as the report miner so the
    two sources merge cleanly.
    """
    if row.get("fidelity") != "F3":
        return None
    obs = row.get("obs")
    if not isinstance(obs, dict):
        return None
    if obs.get("status", row.get("status", "ok")) not in ("ok", "mock"):
        return None
    if obs.get("area_um2") is None:
        return None

    cfg = row.get("config") or {}
    lanes = obs.get("lanes") or cfg.get("mac_lanes") or cfg.get("lanes") or 4
    acc_w = obs.get("acc_w") or cfg.get("accumulator_width") or cfg.get("acc_w") or 24
    clk   = obs.get("clk_ns") or cfg.get("clock_period_ns") or cfg.get("clk_ns")
    abc   = obs.get("effective_abc_recipe") or cfg.get("abc_recipe") or cfg.get("abc")
    platform = row.get("platform") or obs.get("platform") or "nangate45"

    period_ns = obs.get("period_min_ns")
    if period_ns is None and obs.get("fmax_mhz"):
        period_ns = 1000.0 / float(obs["fmax_mhz"])

    return {
        "lanes":   int(lanes),
        "acc_w":   int(acc_w),
        "clk_ns":  float(clk) if clk is not None else None,
        # util/density are frozen at the funnel defaults unless a knob overrode them
        "util":    cfg.get("CORE_UTILIZATION", 40),
        "density": cfg.get("PLACE_DENSITY", 0.60),
        "abc":     abc,
        "platform": platform,
        "area_um2":  obs.get("area_um2"),
        "period_ns": period_ns,
        "fmax_mhz":  obs.get("fmax_mhz"),
        "power_mw":  obs.get("power_mw"),
        "wns_ns":    obs.get("wns_ns"),
        "tns_ns":    obs.get("tns_ns"),
        "setup_viol": obs.get("setup_viol"),
        "timing_met": obs.get("timing_met"),
        "rtl_hash":  _rtl_hash_from_gds(obs.get("gds")),
        "status":    "ok",
    }


def _mine_campaign_rows(design: str | None = None,
                        platform: str | None = None) -> list[dict]:
    """Mine all real F3/ok rows from campaigns/<design>/<platform>/*.jsonl.

    Optional `design`/`platform` filters restrict to one design or PDK (e.g.
    train a clean nangate45-only or gcd-only surrogate).  Each returned row is
    tagged with its provenance ("design", "platform", "variant").
    """
    rows: list[dict] = []
    if not _CAMPAIGN_DIR.is_dir():
        return rows
    for jf in sorted(_CAMPAIGN_DIR.rglob("*.jsonl")):
        parts = jf.relative_to(_CAMPAIGN_DIR).parts
        d = parts[0] if len(parts) >= 1 else "unknown"
        p = parts[1] if len(parts) >= 2 else "unknown"
        if design and d != design:
            continue
        if platform and p != platform:
            continue
        for line in _read(jf).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tr = _campaign_row_to_training(row)
            if tr is None:
                continue
            tr["design"] = d
            tr.setdefault("platform", p)
            tr["variant"] = f"{d}/{jf.stem}"
            rows.append(tr)
    return rows


# ── CSV fallback ───────────────────────────────────────────────────────────────

def _read_sweep_csv(csv_path: Path) -> list[dict]:
    """Read sweep_results.csv rows into flat dicts."""
    import csv
    rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                flat: dict = {}
                for k, v in r.items():
                    try:
                        flat[k] = float(v)
                    except (ValueError, TypeError):
                        flat[k] = v
                # Ensure canonical int types for lanes/acc_w
                for key in ("lanes", "acc_w"):
                    if key in flat:
                        flat[key] = int(flat[key])
                # timing_met: 'YES'/'NO' → bool
                if "timing_met" in flat:
                    flat["timing_met"] = str(flat["timing_met"]).upper() == "YES"
                if "period_ns" not in flat and "fmax_mhz" in flat:
                    fmax = flat.get("fmax_mhz")
                    if fmax and float(fmax) > 0:
                        flat["period_ns"] = 1000.0 / float(fmax)
                flat["status"] = "ok"
                rows.append(flat)
    except (OSError, Exception):
        pass
    return rows


# ── Main data mining ───────────────────────────────────────────────────────────

def _dedup_key(r: dict) -> tuple:
    """Identity of a build for deduplication: design + platform + RTL axes + recipe."""
    return (
        str(r.get("design", "")),
        str(r.get("platform", "nangate45")),
        int(r.get("lanes", 0) or 0),
        int(r.get("acc_w", 0) or 0),
        round(float(r.get("clk_ns", 0.0) or 0.0), 4),
        str(r.get("abc", "") or ""),
    )


def mine_training_rows(design: str | None = None,
                       platform: str | None = None) -> list[dict]:
    """Extract and return all available training rows.

    Sources (in priority order; deduplicated by design/platform/RTL-axes/recipe):
    1. Live campaign JSONL (campaigns/<design>/<platform>/*.jsonl) — the real F3
       data the optimizer now produces. **Primary source.**
    2. Direct report parsing of legacy variant dirs with 6_finish.rpt (usually
       absent — the classic report tree was not retained).
    3. sweep_results*.csv (legacy fallback — usually absent).

    `design`/`platform` optionally restrict to one design or PDK.
    """
    rows: list[dict] = []
    seen: set[tuple] = set()

    # Source 1: live campaign logs (the real terminal-reward data)
    for r in _mine_campaign_rows(design=design, platform=platform):
        if r.get("area_um2") is None:
            continue
        key = _dedup_key(r)
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)

    # Source 2: legacy report variant dirs (nangate45/tinymac only)
    if platform in (None, "nangate45"):
        variant_dirs = sorted(_RPT_DIR.iterdir()) if _RPT_DIR.is_dir() else []
        for vdir in variant_dirs:
            vname = vdir.name
            if vname == "base":
                continue
            parsed = _parse_variant_name(vname)
            if parsed is None:
                continue
            f3 = _extract_f3_metrics(vname)
            if f3 is None:
                continue
            pkey = _proxy_key_for_variant(vname)
            obs = _extract_proxy_obs(pkey)
            row = {**parsed, **f3, **obs,
                   "variant": vname, "platform": "nangate45",
                   "design": "tinymac_accel"}
            key = _dedup_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    # Source 3: sweep CSV (legacy)
    for csv_path in sorted(_MAKE_DIR.glob("sweep_results*.csv")):
        for r in _read_sweep_csv(csv_path):
            r.setdefault("platform", "nangate45")
            r.setdefault("design", "tinymac_accel")
            if r.get("area_um2") is None or int(r.get("lanes", 0) or 0) == 0:
                continue
            key = _dedup_key(r)
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    return rows


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    from collections import Counter

    from gen2.surrogate import Surrogate

    ap = argparse.ArgumentParser(description="Fit + CP3-validate the PPA surrogate.")
    ap.add_argument("--design", default=None,
                    help="restrict training data to one design (e.g. gcd, tinymac_accel)")
    ap.add_argument("--platform", default=None,
                    help="restrict training data to one PDK (e.g. nangate45, asap7)")
    args, _ = ap.parse_known_args()

    print("=" * 65)
    print("fit_surrogate.py — PPA Surrogate CP3 validation")
    print("=" * 65)

    # ── 1. Mine data ──────────────────────────────────────────────────────────
    print("\n[1] Mining training data...")
    if args.design or args.platform:
        print(f"    filter: design={args.design or '*'} platform={args.platform or '*'}")
    rows = mine_training_rows(design=args.design, platform=args.platform)
    print(f"    Found {len(rows)} rows with F3 area_um2 present")
    if len(rows) == 0:
        print("    ERROR: no data found — the classic report tree is absent and no")
        print("    campaign JSONL F3/ok rows matched. Build real F3 data first")
        print("    (see roadmap R8) or relax --design/--platform.")
        sys.exit(1)

    # Provenance breakdown — be explicit about WHAT this surrogate is fit on.
    prov = Counter((r.get("design", "?"), r.get("platform", "?")) for r in rows)
    print("    Provenance (design, platform → rows):")
    for (d, p), n in sorted(prov.items()):
        print(f"      {d:18s} {p:10s}  {n}")
    if len({d for d, _ in prov} ) > 1:
        print("    NOTE: corpus spans multiple designs — the surrogate conflates them")
        print("    via rtl_hash/platform context only. Use --design for a clean fit.")

    # Print a quick summary of what we mined
    lanes_seen = sorted({r["lanes"] for r in rows})
    acc_seen   = sorted({r["acc_w"] for r in rows})
    clk_range  = (min(r["clk_ns"] for r in rows), max(r["clk_ns"] for r in rows))
    area_range = (min(r["area_um2"] for r in rows), max(r["area_um2"] for r in rows))
    print(f"    Lanes seen:  {lanes_seen}")
    print(f"    ACC_W seen:  {acc_seen}")
    print(f"    clk range:   {clk_range[0]:.2f}–{clk_range[1]:.2f} ns")
    print(f"    area range:  {area_range[0]:.0f}–{area_range[1]:.0f} µm²")
    n_with_proxy = sum(1 for r in rows if r.get("proxy_area_um2") is not None)
    n_with_period = sum(1 for r in rows if r.get("period_ns") is not None)
    n_with_power  = sum(1 for r in rows if r.get("power_mw") is not None)
    n_with_ff     = sum(1 for r in rows if r.get("ff_count") is not None)
    print(f"    Rows with proxy_area:  {n_with_proxy}/{len(rows)}")
    print(f"    Rows with period_ns:   {n_with_period}/{len(rows)}")
    print(f"    Rows with power_mw:    {n_with_power}/{len(rows)}")
    print(f"    Rows with ff_count:    {n_with_ff}/{len(rows)}")

    # ── 2. Fit ────────────────────────────────────────────────────────────────
    print("\n[2] Fitting Surrogate (GBT quantile ensemble, 5-fold CV)...")
    s = Surrogate(seed=42)
    diag = s.fit(rows)

    print(f"\n    n_f3 (training rows): {diag['n_f3']}")
    print(f"    n_obs (F2-only rows joined): {diag['n_obs']}")
    if diag.get("fallback"):
        print("    WARNING: fallback mode (too few rows)")

    # ── 3. CP3 validation ─────────────────────────────────────────────────────
    print("\n[3] CP3 validation — 5-fold Spearman ρ:")
    CP3_THRESHOLDS = {"area_um2": 0.80, "period_ns": 0.70, "power_mw": None}
    all_pass = True
    for m in Surrogate.METRICS:
        rho = diag.get(f"cv_rho_{m}", float("nan"))
        n   = diag.get(f"cv_n_{m}", 0)
        thresh = CP3_THRESHOLDS[m]
        if thresh is not None:
            status = "PASS" if (not math.isnan(rho) and rho >= thresh) else "FAIL"
            if status == "FAIL":
                all_pass = False
            flag = f" ← CP3 {'>=' if status == 'PASS' else '<'} {thresh}"
        else:
            status = "INFO"
            flag = " ← informational (power is activity-factor-fiction)"
        print(f"    {m:15s}: ρ = {rho:+.4f}  (n={n})  [{status}]{flag}")

    if all_pass:
        print("\n    ✓  All CP3 thresholds met.")
    else:
        print("\n    ✗  Some CP3 thresholds NOT met — see notes below.")

    # ── 4. Sanity predictions ─────────────────────────────────────────────────
    print("\n[4] Sanity predictions:")

    # a) area(L1) < area(L32) — should always hold (LANES is the #1 area driver)
    pred_L1  = s.predict({"lanes": 1,  "acc_w": 24, "clk_ns": 5.0})
    pred_L32 = s.predict({"lanes": 32, "acc_w": 24, "clk_ns": 5.0})
    mu_L1,  sig_L1  = pred_L1["area_um2"]
    mu_L32, sig_L32 = pred_L32["area_um2"]
    sane_area = mu_L1 < mu_L32
    print(f"\n    area(L1,  A24, clk5) = {mu_L1:.0f} ± {sig_L1:.0f} µm²")
    print(f"    area(L32, A24, clk5) = {mu_L32:.0f} ± {sig_L32:.0f} µm²")
    print(f"    area(L1) < area(L32): {'PASS' if sane_area else 'FAIL'}")

    # b) sigma larger for OOD config (asap7 platform → never seen)
    pred_ood = s.predict({"lanes": 4, "acc_w": 24, "clk_ns": 0.6, "platform": "asap7"})
    pred_ind = s.predict({"lanes": 4, "acc_w": 24, "clk_ns": 5.0, "platform": "nangate45"})
    mu_ood_a,  sig_ood_a  = pred_ood["area_um2"]
    mu_ind_a,  sig_ind_a  = pred_ind["area_um2"]
    # Note: sigma OOD vs in-dist comparison is heuristic for GBT
    # (GBT quantile intervals don't widen via kernel — they extrapolate).
    # We report both and note the limitation.
    print(f"\n    area(asap7, clk=0.6) = {mu_ood_a:.0f} ± {sig_ood_a:.0f} µm²")
    print(f"    area(n45,   clk=5.0) = {mu_ind_a:.0f} ± {sig_ind_a:.0f} µm²")
    print(f"    (sigma OOD vs in-dist: {sig_ood_a:.0f} vs {sig_ind_a:.0f})")
    print(f"    Note: GBT quantile intervals do not widen via kernel; OOD")
    print(f"    uncertainty must be flagged at the architecture level, not here.")

    # c) Reward stats for two reference configs
    pred_r_opt  = s.predict_reward_stats({"lanes": 4,  "acc_w": 24, "clk_ns": 5.0})
    pred_r_slow = s.predict_reward_stats({"lanes": 1,  "acc_w": 24, "clk_ns": 10.0})
    print(f"\n    reward(L4,  A24, clk5)  = {pred_r_opt[0]:+.3f} ± {pred_r_opt[1]:.3f}")
    print(f"    reward(L1,  A24, clk10) = {pred_r_slow[0]:+.3f} ± {pred_r_slow[1]:.3f}")
    print(f"    reward(opt) > reward(slow): {'PASS' if pred_r_opt[0] > pred_r_slow[0] else 'FAIL'}")

    # d) F2-conditioned prediction vs x-only for a variant with known proxy
    # Pick L4_A24_c2p0 which has both proxy and F3 data
    obs_sample = {"proxy_area_um2": 14456.0, "proxy_wns_ns": -4.83}
    pred_xonly = s.predict({"lanes": 4, "acc_w": 24, "clk_ns": 2.0})
    pred_xobs  = s.predict({"lanes": 4, "acc_w": 24, "clk_ns": 2.0}, obs=obs_sample)
    mu_xo,  sig_xo  = pred_xonly["area_um2"]
    mu_xob, sig_xob = pred_xobs["area_um2"]
    print(f"\n    area(L4,A24,c2 | x only) = {mu_xo:.0f} ± {sig_xo:.0f} µm²")
    print(f"    area(L4,A24,c2 | +obs)   = {mu_xob:.0f} ± {sig_xob:.0f} µm²")
    print(f"    (known F3 value: 19738 µm²; proxy shifted sigma: {abs(sig_xo - sig_xob):.0f})")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    print(f"\n[5] Saving model to {_OUT_MODEL} ...")
    _OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    s.save(_OUT_MODEL)
    size_kb = _OUT_MODEL.stat().st_size / 1024
    print(f"    Saved ({size_kb:.1f} KB)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  Training rows used:   {diag['n_f3']}")
    print(f"  With proxy obs:       {n_with_proxy}")
    print(f"  area   CV ρ = {diag.get('cv_rho_area_um2', float('nan')):+.4f}  (threshold ≥ 0.80)")
    print(f"  period CV ρ = {diag.get('cv_rho_period_ns', float('nan')):+.4f}  (threshold ≥ 0.70)")
    print(f"  power  CV ρ = {diag.get('cv_rho_power_mw', float('nan')):+.4f}  (informational)")
    print(f"  Model saved:          {_OUT_MODEL}")
    print(f"  CP3 status:           {'PASS' if all_pass else 'FAIL (see notes above)'}")
    print()

    # CP3 notes if fail
    if not all_pass:
        print("  CP3 remedies per doc:")
        rho_area = diag.get("cv_rho_area_um2", float("nan"))
        rho_period = diag.get("cv_rho_period_ns", float("nan"))
        if math.isnan(rho_area) or rho_area < 0.80:
            print("  - area < 0.80: add more F3 rows or F2 proxy obs to increase n")
        if math.isnan(rho_period) or rho_period < 0.70:
            print("  - period < 0.70: expected — period tracks the CONSTRAINT (effort")
            print("    coupling); use proxy WNS as the conditioning observable.")
            print("    Rho within matched-clk groups will be higher.")


if __name__ == "__main__":
    main()

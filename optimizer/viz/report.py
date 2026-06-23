#!/usr/bin/env python3
"""report.py — static self-contained HTML report for a funnel-optimizer campaign.

    python3 optimizer/viz/report.py                       # latest campaign
    python3 optimizer/viz/report.py --campaign all
    python3 optimizer/viz/report.py --log tinymac_accel_run1.jsonl --open
    python3 optimizer/viz/report.py --out /tmp/run.html

Produces one HTML file (Plotly CDN, no server) with:
  - optimization history: per-episode reward + best-so-far, vs episode and vs
    wall-clock hours
  - reward vs each parameter (scatter, coloured by fidelity; box plot for
    categorical params)
  - fidelity funnel (how many configs died at F0/F1/F2 vs reached F3)
  - reward distribution
  - Optuna param-importances, slice and contour plots over the F3 trials
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viz.campaign_data import (  # noqa: E402
    DEFAULT_LOG,
    CampaignData,
    build_study,
    episode_value,
    obs_objective,
)

_FIDELITY_COLORS = {
    "F0": "#d62728",   # red — died at proxy gate 0
    "F1": "#ff7f0e",
    "F2": "#9467bd",
    "F3": "#2ca02c",   # green — reached full flow
    None: "#7f7f7f",
}

# Asap7 baseline: first documented GDS (hand-picked, Stage 6, CLAUDE.md)
_ASAP7_BASELINE = {
    "area_um2": 1433,
    "fmax_mhz": 509,
    "wns_ns": -0.96,
    "timing_met": False,
    "config": {"mac_lanes": 4, "accumulator_width": 24, "clock_period_ns": 1.0, "abc_recipe": "orfs_speed"},
    "label": "Baseline (L4_A24 @ 1.0 ns)",
}

# Qualitative palette for mac_lanes values
_LANES_PALETTE = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c", 8: "#d62728", 16: "#9467bd", 32: "#8c564b"}


def _running_max(xs):
    best = float("-inf")
    out = []
    for x in xs:
        if x is not None and x > best:
            best = x
        out.append(best if best != float("-inf") else None)
    return out


def build_figures(data: CampaignData):
    """Return a list of (section_title, plotly Figure)."""
    import plotly.graph_objects as go

    rows = data.rows
    figs: list[tuple[str, object]] = []

    episodes = [r.get("episode", i + 1) for i, r in enumerate(rows)]
    hours = [(r.get("spent_s") or 0.0) / 3600.0 for r in rows]
    vals = [episode_value(r) for r in rows]
    fids = [r.get("fidelity") for r in rows]
    best = _running_max(vals)

    # ── optimization history vs episode ──────────────────────────────────────
    fig = go.Figure()
    for fid in ["F0", "F1", "F2", "F3"]:
        xs = [e for e, f in zip(episodes, fids) if f == fid]
        ys = [v for v, f in zip(vals, fids) if f == fid]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=f"reward ({fid})",
                marker=dict(color=_FIDELITY_COLORS[fid], size=7, opacity=0.75),
            ))
    fig.add_trace(go.Scatter(
        x=episodes, y=best, mode="lines", name="best-so-far",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.update_layout(title="Optimization history (reward vs episode)",
                      xaxis_title="episode", yaxis_title="reward (real_speedup)")
    figs.append(("history-episode", fig))

    # ── optimization history vs wall-clock ───────────────────────────────────
    if any(h > 0 for h in hours):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=vals, mode="markers", name="reward",
            marker=dict(color=[_FIDELITY_COLORS.get(f) for f in fids], size=7,
                        opacity=0.75),
        ))
        fig.add_trace(go.Scatter(x=hours, y=best, mode="lines", name="best-so-far",
                                 line=dict(color="#1f77b4", width=2)))
        fig.update_layout(title="Reward vs wall-clock budget spent",
                          xaxis_title="hours spent", yaxis_title="reward")
        figs.append(("history-time", fig))

    # ── reward vs each parameter ─────────────────────────────────────────────
    for name, spec in data.specs.items():
        xs_all, ys_all, c_all = [], [], []
        for r, v in zip(rows, vals):
            cfg = r.get("config") or {}
            if name not in cfg or cfg[name] is None or v is None:
                continue
            xs_all.append(cfg[name])
            ys_all.append(v)
            c_all.append(r.get("fidelity"))
        if not xs_all:
            continue
        fig = go.Figure()
        if spec.kind == "cat":
            for choice in spec.choices:
                ys = [y for x, y in zip(xs_all, ys_all) if str(x) == choice]
                if ys:
                    fig.add_trace(go.Box(y=ys, name=choice, boxpoints="all",
                                         jitter=0.4, pointpos=0))
            fig.update_layout(title=f"reward by {name}", xaxis_title=name,
                              yaxis_title="reward", showlegend=False)
        else:
            for fid in ["F0", "F1", "F2", "F3"]:
                xs = [x for x, c in zip(xs_all, c_all) if c == fid]
                ys = [y for y, c in zip(ys_all, c_all) if c == fid]
                if xs:
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="markers", name=fid,
                        marker=dict(color=_FIDELITY_COLORS[fid], size=7,
                                    opacity=0.75)))
            fig.update_layout(title=f"reward vs {name}", xaxis_title=name,
                              yaxis_title="reward")
        figs.append((f"param-{name}", fig))

    # ── fidelity funnel ──────────────────────────────────────────────────────
    counts = data.fidelity_counts()
    fig = go.Figure(go.Bar(
        x=list(counts.keys()), y=list(counts.values()),
        marker_color=[_FIDELITY_COLORS[f] for f in counts.keys()],
        text=list(counts.values()), textposition="outside",
    ))
    fig.update_layout(title="Fidelity funnel (episodes ending at each gate)",
                      xaxis_title="fidelity reached", yaxis_title="episodes")
    figs.append(("funnel", fig))

    # ── reward distribution (F3 only — real terminal rewards) ────────────────
    f3_vals = [v for v, f in zip(vals, fids) if f == "F3" and v is not None]
    if f3_vals:
        fig = go.Figure(go.Histogram(x=f3_vals, nbinsx=30,
                                     marker_color=_FIDELITY_COLORS["F3"]))
        fig.update_layout(title="F3 terminal-reward distribution",
                          xaxis_title="reward", yaxis_title="count")
        figs.append(("dist", fig))

    return figs


def build_optuna_figures(rows):
    """Optuna importance / slice / contour over reconstructed trials (best-effort)."""
    import optuna.visualization as vis

    figs: list[tuple[str, object]] = []
    try:
        study, _added, specs = build_study(rows, study_name="report")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not reconstruct Optuna study: {exc}")
        return figs
    if len(study.trials) < 4:
        return figs

    def _try(label, fn):
        try:
            figs.append((label, fn()))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] optuna {label} skipped: {exc}")

    _try("optuna-importances", lambda: vis.plot_param_importances(study))
    _try("optuna-slice", lambda: vis.plot_slice(study))
    _try("optuna-parallel", lambda: vis.plot_parallel_coordinate(study))
    # contour over the two most important numeric params, if available
    numeric = [n for n, s in specs.items() if s.kind in ("int", "float")]
    if len(numeric) >= 2:
        _try("optuna-contour", lambda: vis.plot_contour(study, params=numeric[:2]))
    return figs


def _f3_ok_rows(rows: list[dict]) -> list[dict]:
    """Return rows with fidelity=F3 and status=ok that have obs metrics."""
    return [
        r for r in rows
        if r.get("fidelity") == "F3" and r.get("status") == "ok"
        and r.get("obs", {}).get("area_um2") is not None
        and r.get("obs", {}).get("fmax_mhz") is not None
    ]


def _pareto_front(pts: list[tuple]) -> list[tuple]:
    """Non-dominated set over (area, fmax): minimize area, maximize fmax."""
    front = []
    for a, f, r in pts:
        dominated = any(
            a2 <= a and f2 >= f and (a2 < a or f2 > f)
            for a2, f2, _ in pts
        )
        if not dominated:
            front.append((a, f, r))
    return sorted(front)  # ascending area


def build_pareto_figure(rows: list[dict]) -> tuple[str, object]:
    """Area vs Fmax scatter for all F3 results, with Pareto frontier curve and baseline."""
    import plotly.graph_objects as go

    f3 = _f3_ok_rows(rows)
    fig = go.Figure()

    if not f3:
        fig.add_annotation(text="No F3 data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18))
        fig.update_layout(title="Pareto Frontier: Area vs Fmax")
        return ("pareto", fig)

    pts = [(r["obs"]["area_um2"], r["obs"]["fmax_mhz"], r) for r in f3]

    # One trace per mac_lanes value (grouped for legend)
    lanes_vals = sorted({r["config"].get("mac_lanes", 0) for r in f3})
    for lanes in lanes_vals:
        subset = [(a, fmax, r) for a, fmax, r in pts if r["config"].get("mac_lanes") == lanes]
        if not subset:
            continue
        hover = [
            f"L{r['config'].get('mac_lanes')}_A{r['config'].get('accumulator_width')}<br>"
            f"clk={r['config'].get('clock_period_ns', 0):.2f} ns<br>"
            f"recipe={r['config'].get('abc_recipe')}<br>"
            f"area={r['obs']['area_um2']:.0f} µm²<br>"
            f"fmax={r['obs']['fmax_mhz']:.0f} MHz<br>"
            f"wns={r['obs'].get('wns_ns', 0):.3f} ns<br>"
            f"power={r['obs'].get('power_mw', 0):.0f} mW"
            for _, _, r in subset
        ]
        # Marker size scaled by power_mw (normalized 8–20px range)
        powers = [r["obs"].get("power_mw") or 250 for _, _, r in subset]
        p_min, p_max = min(powers), max(powers)
        p_range = max(p_max - p_min, 1)
        sizes = [8 + 12 * (p - p_min) / p_range for p in powers]
        fig.add_trace(go.Scatter(
            x=[a for a, _, _ in subset],
            y=[fmax for _, fmax, _ in subset],
            mode="markers",
            name=f"L={lanes}",
            marker=dict(color=_LANES_PALETTE.get(lanes, "#aaa"), size=sizes,
                        opacity=0.8, line=dict(width=1, color="white")),
            hovertext=hover, hoverinfo="text",
        ))

    # Pareto frontier step-line
    front = _pareto_front(pts)
    if front:
        # Extend steps for a staircase look
        px_step = [front[0][0]] + [a for a, _, _ in front]
        py_step = [f for _, f, _ in front] + [front[-1][1]]
        fig.add_trace(go.Scatter(
            x=px_step, y=py_step, mode="lines",
            name="Pareto frontier",
            line=dict(color="#1f2937", width=2, dash="dash"),
            showlegend=True,
        ))

    # Baseline marker
    b = _ASAP7_BASELINE
    fig.add_trace(go.Scatter(
        x=[b["area_um2"]], y=[b["fmax_mhz"]],
        mode="markers+text",
        name=b["label"],
        marker=dict(color="red", size=16, symbol="star",
                    line=dict(width=2, color="darkred")),
        text=[b["label"]],
        textposition="top right",
        hovertext=f"{b['label']}<br>area={b['area_um2']} µm²<br>fmax={b['fmax_mhz']} MHz<br>wns={b['wns_ns']} ns",
        hoverinfo="text",
    ))

    fig.update_layout(
        title="Pareto Frontier: Area vs Fmax (asap7) — marker size = power",
        xaxis_title="Area (µm²)",
        yaxis_title="Fmax (MHz)",
        legend=dict(title="mac_lanes"),
    )
    return ("pareto", fig)


def build_iteration_figure(rows: list[dict]) -> tuple[str, object]:
    """Fmax progression across F3 runs in chronological order."""
    import plotly.graph_objects as go

    f3 = sorted(_f3_ok_rows(rows), key=lambda r: r.get("ts", 0))
    fig = go.Figure()

    if not f3:
        fig.add_annotation(text="No F3 data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18))
        fig.update_layout(title="Fmax Progression Over F3 Runs")
        return ("iteration", fig)

    run_nums = list(range(1, len(f3) + 1))
    fmaxes = [r["obs"]["fmax_mhz"] for r in f3]
    lanes_colors = [_LANES_PALETTE.get(r["config"].get("mac_lanes", 0), "#aaa") for r in f3]

    hover = [
        f"Run #{i}<br>"
        f"L{r['config'].get('mac_lanes')}_A{r['config'].get('accumulator_width')}<br>"
        f"clk={r['config'].get('clock_period_ns', 0):.2f} ns  {r['config'].get('abc_recipe')}<br>"
        f"fmax={r['obs']['fmax_mhz']:.0f} MHz  area={r['obs']['area_um2']:.0f} µm²"
        for i, r in zip(run_nums, f3)
    ]

    # All F3 scatter points (colored by lanes)
    for lanes in sorted({r["config"].get("mac_lanes", 0) for r in f3}):
        xs = [n for n, r in zip(run_nums, f3) if r["config"].get("mac_lanes") == lanes]
        ys = [r["obs"]["fmax_mhz"] for r in f3 if r["config"].get("mac_lanes") == lanes]
        ht = [h for h, r in zip(hover, f3) if r["config"].get("mac_lanes") == lanes]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=f"L={lanes}",
                marker=dict(color=_LANES_PALETTE.get(lanes, "#aaa"), size=9, opacity=0.85),
                hovertext=ht, hoverinfo="text",
            ))

    # Best-so-far line
    best_so_far = _running_max(fmaxes)
    fig.add_trace(go.Scatter(
        x=run_nums, y=best_so_far, mode="lines", name="best-so-far",
        line=dict(color="#1f2937", width=2),
    ))

    # Annotate new-best points
    cur_best = float("-inf")
    for i, (fmax, r) in enumerate(zip(fmaxes, f3)):
        if fmax > cur_best:
            cur_best = fmax
            cfg = r["config"]
            label = f"L{cfg.get('mac_lanes')}_A{cfg.get('accumulator_width')}<br>{fmax:.0f} MHz"
            fig.add_annotation(
                x=run_nums[i], y=fmax,
                text=label, showarrow=True, arrowhead=2,
                ax=30, ay=-35, font=dict(size=10, color="#1f2937"),
            )

    # Vertical dashed line where L8 cluster begins (first run where L8 dominates remaining)
    lanes_seq = [r["config"].get("mac_lanes", 0) for r in f3]
    # Find the first run where >=60% of the remaining runs are L8
    l8_start = None
    for i in range(len(lanes_seq)):
        remaining = lanes_seq[i:]
        if len(remaining) >= 5 and remaining.count(8) / len(remaining) >= 0.6:
            l8_start = run_nums[i]
            break
    if l8_start is not None:
        fig.add_vline(x=l8_start, line=dict(color="#9467bd", width=1.5, dash="dot"),
                      annotation_text="L8 cluster", annotation_position="top right",
                      annotation_font=dict(color="#9467bd"))

    # Baseline reference line
    fig.add_hline(y=_ASAP7_BASELINE["fmax_mhz"],
                  line=dict(color="red", width=1.5, dash="dash"),
                  annotation_text=f"Baseline {_ASAP7_BASELINE['fmax_mhz']} MHz",
                  annotation_position="bottom right",
                  annotation_font=dict(color="red"))

    fig.update_layout(
        title=f"Fmax Progression Over {len(f3)} Full P&R Runs (asap7, chronological)",
        xaxis_title="Run # (chronological by timestamp)",
        yaxis_title="Fmax (MHz)",
        legend=dict(title="mac_lanes"),
    )
    return ("iteration", fig)


def build_comparison_table(rows: list[dict]) -> tuple[str, object]:
    """Before/after table: baseline vs Pareto-optimal configs found by optimizer."""
    import plotly.graph_objects as go

    f3 = _f3_ok_rows(rows)
    pts = [(r["obs"]["area_um2"], r["obs"]["fmax_mhz"], r) for r in f3] if f3 else []
    pareto = _pareto_front(pts)  # sorted by ascending area

    b = _ASAP7_BASELINE
    b_cfg = b["config"]

    headers = ["Config", "Lanes", "Acc_W", "Clock (ns)", "Area (µm²)",
               "Fmax (MHz)", "WNS (ns)", "Recipe", "ΔFmax", "ΔArea"]

    def _delta(val, base, higher_better=True):
        pct = (val - base) / base * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.1f}%"

    # Build rows: first the baseline, then each Pareto config
    col_config, col_lanes, col_accw, col_clk = [], [], [], []
    col_area, col_fmax, col_wns, col_recipe = [], [], [], []
    col_dfmax, col_darea = [], []
    row_colors = []

    # Baseline row
    col_config.append(b["label"])
    col_lanes.append(b_cfg["mac_lanes"])
    col_accw.append(b_cfg["accumulator_width"])
    col_clk.append(f"{b_cfg['clock_period_ns']:.2f}")
    col_area.append(f"{b['area_um2']}")
    col_fmax.append(f"{b['fmax_mhz']}")
    col_wns.append(f"{b['wns_ns']:.3f}")
    col_recipe.append(b_cfg["abc_recipe"])
    col_dfmax.append("—")
    col_darea.append("—")
    row_colors.append("#fca5a5")  # light red for baseline

    for area, fmax, r in pareto:
        cfg = r["config"]
        obs = r["obs"]
        lbl = f"L{cfg.get('mac_lanes')}_A{cfg.get('accumulator_width')} @ {cfg.get('clock_period_ns', 0):.2f} ns"
        col_config.append(lbl)
        col_lanes.append(cfg.get("mac_lanes", ""))
        col_accw.append(cfg.get("accumulator_width", ""))
        col_clk.append(f"{cfg.get('clock_period_ns', 0):.3f}")
        col_area.append(f"{area:.0f}")
        col_fmax.append(f"{fmax:.0f}")
        col_wns.append(f"{obs.get('wns_ns', 0):.3f}")
        col_recipe.append(obs.get("effective_abc_recipe") or cfg.get("abc_recipe", ""))
        col_dfmax.append(_delta(fmax, b["fmax_mhz"]))
        col_darea.append(_delta(area, b["area_um2"], higher_better=False))
        row_colors.append("#d1fae5" if fmax > b["fmax_mhz"] else "#fef9c3")

    # Transpose for Plotly (it wants column-major)
    all_cols = [col_config, col_lanes, col_accw, col_clk, col_area,
                col_fmax, col_wns, col_recipe, col_dfmax, col_darea]

    fig = go.Figure(go.Table(
        header=dict(
            values=headers,
            fill_color="#1e3a5f",
            font=dict(color="white", size=13),
            align="center",
            height=36,
        ),
        cells=dict(
            values=all_cols,
            fill_color=[row_colors] * len(headers),
            align=["left"] + ["center"] * (len(headers) - 1),
            font=dict(size=12),
            height=30,
        ),
    ))
    fig.update_layout(
        title=dict(text="Before / After: Baseline vs Optimizer Pareto Frontier (asap7)", font=dict(size=15)),
        margin=dict(t=48, b=8, l=8, r=8),
    )
    return ("full-comparison", fig)


def build_knob_tier_figure() -> tuple[str, object]:
    """Static table of all 18 search variables organized by importance tier."""
    import plotly.graph_objects as go

    tiers = [
        ("Tier 1 · RTL & Timing", "#bfdbfe",
         [
             ("mac_lanes",         "int",   "{1, 2, 4, 8, 16, 32}"),
             ("accumulator_width", "int",   "{16, 24, 32}"),
             ("clock_period_ns",   "float", "0.5 – 1.9 ns (asap7)"),
             ("abc_recipe",        "cat",   "{orfs_speed, orfs_area, plain}"),
         ]),
        ("Tier 2 · Floorplan", "#a7f3d0",
         [
             ("CORE_ASPECT_RATIO",                  "float", "0.5 – 3.0"),
             ("CORE_MARGIN",                        "float", "1.0 – 5.0 µm"),
             ("PLACE_DENSITY_LB_ADDON",             "float", "0.0 – 0.15"),
             ("CELL_PAD_IN_SITES_GLOBAL_PLACEMENT", "int",   "0 – 4 sites"),
             ("CELL_PAD_IN_SITES_DETAIL_PLACEMENT", "int",   "0 – 4 sites"),
         ]),
        ("Tier 3 · CTS / Route", "#ddd6fe",
         [
             ("CTS_CLUSTER_SIZE",             "int",   "30 – 200"),
             ("CTS_CLUSTER_DIAMETER",         "float", "100 – 500 µm"),
             ("TNS_END_PERCENT",              "float", "0 – 100 %"),
             ("SETUP_SLACK_MARGIN",           "float", "0.0 – 0.5 ns"),
             ("ROUTING_LAYER_ADJUSTMENT",     "float", "0.0 – 0.7"),
             ("RECOVER_POWER",                "int",   "0 or 1"),
             ("DETAILED_ROUTE_END_ITERATION", "int",   "1 – 10"),
         ]),
        ("Tier 3 · Placement step", "#fde68a",
         [
             ("MIN_PLACE_STEP_COEF", "float", "0.1 – 1.0"),
             ("MAX_PLACE_STEP_COEF", "float", "0.5 – 2.0"),
         ]),
    ]

    col_tier, col_var, col_type, col_range, row_colors = [], [], [], [], []
    for tier_name, color, knobs in tiers:
        for var, typ, rng in knobs:
            col_tier.append(tier_name)
            col_var.append(var)
            col_type.append(typ)
            col_range.append(rng)
            row_colors.append(color)

    fig = go.Figure(go.Table(
        header=dict(
            values=["Tier", "Variable", "Type", "Range / Choices"],
            fill_color="#1f2937",
            font=dict(color="white", size=13),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[col_tier, col_var, col_type, col_range],
            fill_color=[row_colors] * 4,
            align="left",
            font=dict(size=12, family="monospace"),
            height=28,
        ),
    ))
    fig.update_layout(
        title=dict(text="18 Search Variables — 4 Importance Tiers", font=dict(size=15)),
        margin=dict(t=48, b=8, l=8, r=8),
    )
    return ("full-knob-tiers", fig)


def build_speedup_figure(rows: list[dict]) -> tuple[str, object]:
    """Area vs real-speedup-at-Fmax scatter for all F3 results.

    Real speedup = SW_LATENCY / accel_latency_at_fmax, where
      SW_LATENCY  = 11,196,638 cycles × 10 ns  (100 MHz PicoRV32, no accel)
      accel_latency = AVG_CYCLES[lanes] × (1000 / fmax_mhz) ns
    """
    import plotly.graph_objects as go

    _SW_LATENCY_NS = 11_196_638 * 10.0
    _AVG_CYCLES = {1: 273_130, 2: 152_140, 4: 91_650, 8: 61_400, 16: 46_670, 32: 39_310}

    f3 = _f3_ok_rows(rows)
    fig = go.Figure()

    if not f3:
        fig.add_annotation(text="No F3 data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18))
        fig.update_layout(title="Area vs Real Speedup")
        return ("speedup", fig)

    for lanes in sorted({r["config"].get("mac_lanes", 0) for r in f3}):
        subset = [r for r in f3 if r["config"].get("mac_lanes") == lanes]
        acyc = _AVG_CYCLES.get(lanes, 91_650)
        xs, ys, hover = [], [], []
        for r in subset:
            fmax = r["obs"]["fmax_mhz"]
            area = r["obs"]["area_um2"]
            period_ns = 1000.0 / fmax
            speedup = _SW_LATENCY_NS / (acyc * period_ns)
            xs.append(area)
            ys.append(speedup)
            hover.append(
                f"L{r['config'].get('mac_lanes')}_A{r['config'].get('accumulator_width')}<br>"
                f"clk={r['config'].get('clock_period_ns', 0):.2f} ns · {r['config'].get('abc_recipe')}<br>"
                f"Fmax={fmax:.0f} MHz · area={area:.0f} µm²<br>"
                f"<b>Speedup = {speedup:.0f}×</b>"
            )
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=f"L={lanes}",
            marker=dict(color=_LANES_PALETTE.get(lanes, "#aaa"), size=10,
                        opacity=0.85, line=dict(width=1, color="white")),
            hovertext=hover, hoverinfo="text",
        ))

    # Baseline marker
    b = _ASAP7_BASELINE
    b_lanes = b["config"]["mac_lanes"]
    b_fmax = b["fmax_mhz"]
    b_area = b["area_um2"]
    b_speedup = _SW_LATENCY_NS / (_AVG_CYCLES.get(b_lanes, 91_650) * (1000.0 / b_fmax))
    fig.add_trace(go.Scatter(
        x=[b_area], y=[b_speedup],
        mode="markers+text", name=b["label"],
        marker=dict(color="red", size=16, symbol="star",
                    line=dict(width=2, color="darkred")),
        text=["Baseline"], textposition="top right",
        hovertext=f"{b['label']}<br>area={b_area} µm² · fmax={b_fmax} MHz<br>Speedup={b_speedup:.0f}×",
        hoverinfo="text",
    ))

    fig.update_layout(
        title=dict(text="Area vs Inference Speedup (at Fmax) — vs SW Baseline (112 ms @ 100 MHz)", font=dict(size=14)),
        xaxis_title="Area (µm²)",
        yaxis_title="Speedup over SW baseline",
        legend=dict(title="mac_lanes"),
        annotations=[dict(
            text="<i>Higher-left = smaller chip, faster inference</i>",
            xref="paper", yref="paper", x=0.01, y=0.99,
            showarrow=False, font=dict(size=11, color="#6b7280"),
            align="left",
        )],
    )
    return ("speedup", fig)


_CSS = """
body {
  font-family: system-ui, -apple-system, sans-serif;
  margin: 0;
  background: #f1f5f9;
  color: #1e293b;
}
header {
  background: linear-gradient(135deg, #1e3a5f 0%, #1f2937 100%);
  color: #fff;
  padding: 20px 28px 16px;
}
header h1 {
  margin: 0 0 4px;
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.3px;
}
header p.subtitle {
  margin: 0;
  font-size: 13px;
  color: #94a3b8;
}
.banner {
  display: flex;
  gap: 12px;
  padding: 16px 16px 0;
  flex-wrap: wrap;
}
.kpi {
  flex: 1;
  min-width: 160px;
  background: #fff;
  border-radius: 10px;
  border: 1px solid #e2e8f0;
  padding: 14px 18px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.kpi-val {
  font-size: 26px;
  font-weight: 700;
  color: #1e3a5f;
  line-height: 1.1;
}
.kpi-val.green { color: #059669; }
.kpi-val.amber { color: #d97706; }
.kpi-label {
  font-size: 12px;
  color: #64748b;
  margin-top: 4px;
}
.section-header {
  grid-column: 1 / -1;
  padding: 10px 4px 2px;
  font-size: 13px;
  font-weight: 600;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  border-bottom: 1px solid #cbd5e1;
  margin-top: 8px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
  gap: 12px;
  padding: 16px;
}
.card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.card.full { grid-column: 1 / -1; }
"""


def build_summary_banner(rows: list[dict]) -> tuple[str, str]:
    """Return a raw-HTML KPI banner (label='--html:banner', fig=html_string)."""
    _SW_LATENCY_NS = 11_196_638 * 10.0
    _AVG_CYCLES = {1: 273_130, 2: 152_140, 4: 91_650, 8: 61_400, 16: 46_670, 32: 39_310}

    f3 = _f3_ok_rows(rows)
    b = _ASAP7_BASELINE

    if not f3:
        return ("--html:banner", "<div class='banner'><div class='kpi'><div class='kpi-val'>No F3 data</div></div></div>")

    best_fmax = max(r["obs"]["fmax_mhz"] for r in f3)
    min_area  = min(r["obs"]["area_um2"]  for r in f3)

    # Best speedup across all F3 results
    best_speedup = 0.0
    for r in f3:
        lanes = r["config"].get("mac_lanes", 4)
        fmax  = r["obs"]["fmax_mhz"]
        sp = _SW_LATENCY_NS / (_AVG_CYCLES.get(lanes, 91_650) * (1000.0 / fmax))
        best_speedup = max(best_speedup, sp)

    total_h = sum(r.get("cost_s", 0) for r in rows if r.get("fidelity") == "F3") / 3600
    n_vars  = len({k for r in rows for k in (r.get("config") or {})})

    fmax_delta = (best_fmax - b["fmax_mhz"]) / b["fmax_mhz"] * 100
    area_delta = (min_area  - b["area_um2"])  / b["area_um2"]  * 100

    html = f"""<div class='banner'>
  <div class='kpi'>
    <div class='kpi-val green'>{best_fmax:.0f} MHz</div>
    <div class='kpi-label'>Best Fmax found &nbsp;·&nbsp; <b>+{fmax_delta:.1f}%</b> vs hand-picked baseline</div>
  </div>
  <div class='kpi'>
    <div class='kpi-val green'>{best_speedup:.0f}×</div>
    <div class='kpi-label'>Peak inference speedup vs software baseline (112 ms @ 100 MHz)</div>
  </div>
  <div class='kpi'>
    <div class='kpi-val'>{min_area:.0f} µm²</div>
    <div class='kpi-label'>Minimum area found &nbsp;·&nbsp; <b>{area_delta:+.1f}%</b> vs baseline</div>
  </div>
  <div class='kpi'>
    <div class='kpi-val amber'>{len(f3)} builds</div>
    <div class='kpi-label'>{len(f3)} full P&amp;R runs in {total_h:.1f} h &nbsp;·&nbsp; {n_vars} variables searched</div>
  </div>
</div>"""
    return ("--html:banner", html)


def write_html(figs, out_path: Path, title: str, subtitle: str = ""):
    """Write a self-contained HTML report.

    ``figs`` is a list of ``(label, content)`` where:
      - label = "section:Title"      → full-width section header (content ignored)
      - label = "--html:..."         → content is a raw HTML string, injected directly
      - label = "full-..."           → card spans the full grid width
      - otherwise                    → standard half-width card with Plotly figure
    """
    parts = [
        "<!doctype html><html><head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        f"<header><h1>{title}</h1>",
        (f"<p class='subtitle'>{subtitle}</p>" if subtitle else ""),
        "</header>",
        "<div class='grid'>",
    ]
    plotlyjs_included = False
    for label, content in figs:
        if label.startswith("section:"):
            heading = label[len("section:"):]
            parts.append(f"<div class='section-header'>{heading}</div>")
            continue
        if label.startswith("--html:"):
            # Raw HTML block — spans full width
            parts.append(f"<div class='card full'>{content}</div>")
            continue
        full = label.startswith("full-")
        cls = "card full" if full else "card"
        height = "520px" if full else "460px"
        inc_js = "cdn" if not plotlyjs_included else False
        frag = content.to_html(full_html=False, include_plotlyjs=inc_js,
                               default_height=height)
        plotlyjs_included = True
        parts.append(f"<div class='{cls}'>{frag}</div>")
    parts.append("</div></body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Static HTML report for a campaign log")
    ap.add_argument("--log",
                    default=str(DEFAULT_LOG) if DEFAULT_LOG is not None else None,
                    required=(DEFAULT_LOG is None),
                    help="campaign JSONL path (required when no results_funnel_campaigns.jsonl exists)")
    ap.add_argument("--campaign", default="latest",
                    help="campaign_id | 'latest' | 'all'")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--no-optuna", action="store_true",
                    help="skip Optuna importance/slice/contour figures")
    ap.add_argument("--no-supervisor", action="store_true",
                    help="skip supervisor overview section (Pareto, comparison table, etc.)")
    ap.add_argument("--open", action="store_true", help="open the report in a browser")
    args = ap.parse_args()

    data = CampaignData.load(args.log, args.campaign)
    if not data.rows:
        print(f"No episodes found in {args.log} for campaign={args.campaign!r}")
        sys.exit(1)

    f3_count = sum(1 for r in data.rows if r.get("fidelity") == "F3" and r.get("status") == "ok")
    print(f"Loaded {len(data.rows)} episodes (campaign={data.campaign_id}, "
          f"{len(data.specs)} params, {f3_count} F3 results)")

    # ── infer design / platform from the log path for a human-readable title ──
    log_path = Path(args.log)
    platform = log_path.parent.name
    design   = log_path.parent.parent.name
    title    = f"TinyMAC Accelerator · {design} · {platform} · {f3_count} Full P&R Builds"
    subtitle = (f"Multi-fidelity funnel optimizer · {len(data.rows):,} candidate evaluations"
                f" · {len(data.specs)} search variables · campaign {data.campaign_id}")

    # ── assemble figures with section structure ───────────────────────────────
    figs: list[tuple[str, object]] = []

    if not args.no_supervisor:
        figs.append(("section:Supervisor Overview", None))
        figs.append(build_summary_banner(data.rows))
        figs.append(build_comparison_table(data.rows))
        figs.append(build_pareto_figure(data.rows))
        figs.append(build_iteration_figure(data.rows))
        figs.append(build_speedup_figure(data.rows))
        figs.append(build_knob_tier_figure())

    figs.append(("section:Optimization History", None))
    analysis = build_figures(data)
    # Split: history + funnel + dist first, then per-param scatters
    history = [(l, f) for l, f in analysis if not l.startswith("param-")]
    params  = [(l, f) for l, f in analysis if l.startswith("param-")]
    figs.extend(history)

    if params:
        figs.append(("section:Per-Parameter Reward Analysis", None))
        figs.extend(params)

    if not args.no_optuna:
        optuna_figs = build_optuna_figures(data.rows)
        if optuna_figs:
            figs.append(("section:Optuna Analysis", None))
            figs.extend(optuna_figs)

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "reports" / f"report_{data.campaign_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_html(figs, out, title, subtitle)
    printable = [(l, f) for l, f in figs if not l.startswith("section:") and not l.startswith("--html:")]
    print(f"Wrote {len(printable)} figures → {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()

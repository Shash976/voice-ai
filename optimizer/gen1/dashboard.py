"""dashboard.py — live animated dashboard for the TinyMAC design-space explorer.

Layout
------
  Row 0 : title + description
  Row 1 : 7-column KPI bar
  Row 2 : [Chip Floorplan — proportional, animated]  |  [PPA Evolution — 4-panel live]
  Row 3 : Trial History Strip  (full width, one cell per trial)
  Row 4 : [Pareto frontier]  |  [Parallel coordinates]
  Row 5 : Trial table  (collapsible)

The "video" effect
------------------
Every config produces a visually distinct floorplan:
  - mac_lanes 1 → 16  :  MAC array grows from a single tiny cell to a wide 8×2 grid
  - buffer_bytes 256 → 4096  :  SRAM blocks grow taller
  - accumulator_width 16/24/32  :  accumulator bar changes thickness; red if overflow risk
  - timing violation  :  die border turns red

The 2-second auto-refresh drives all animations (MAC-lane sweep, new data points on PPA graphs).

Launch
------
  streamlit run optimizer/dashboard.py

SSH tunnel (company VM, no inbound port):
  ssh -L 8501:localhost:8501 user@vm   →   open  http://localhost:8501  locally

Backward compatible with both old flat records (search.py) and new nested records (run_optimizer.py).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_FILE = Path(__file__).resolve().parent.parent / "results" / "gen1" / "results.jsonl"
# Import SW_BASELINE_CYCLES from the single source of truth (constants.py).
# Exposed as SW_BASELINE for backward compatibility with callers in this file.
try:
    from common.constants import SW_BASELINE_CYCLES as SW_BASELINE  # type: ignore[import]
except ImportError:
    SW_BASELINE = 11_196_638   # fallback if constants.py unavailable (dashboard import)
REFRESH_SEC  = 2

# Colour palette (dark chip-design aesthetic)
BG_DEEP      = "rgba(7, 9, 16, 1)"
BG_DIE       = "rgba(12, 16, 26, 1)"
BG_PANEL     = "rgba(18, 22, 34, 1)"
GRID_CLR     = "rgba(255,255,255,0.05)"
C_CYAN       = "#00d4ff"
C_ORANGE     = "#ff6b35"
C_GREEN      = "#00ff88"
C_GOLD       = "#ffd700"
C_RED        = "#ff3355"
C_PURPLE     = "#cc55ff"
C_BLUE       = "#3399ff"
C_PINK       = "#ff4499"


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="TinyMAC Live Explorer", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
  .stApp { background:#070910; color:#dce6f0; }
  section[data-testid="stSidebar"] { background:#070910; }
  div[data-testid="metric-container"] {
      background:#10141f; border:1px solid #1e2840;
      border-radius:8px; padding:10px 14px; }
  div[data-testid="metric-container"] label { color:#5a7a9a !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
      color:#00d4ff !important; font-size:1.5rem !important; }
  hr { border-color:#1e2840; }
  .stPlotlyChart { background:#070910; }
</style>""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def _flatten(rec: dict) -> dict:
    """Normalise both old flat records and new nested records."""
    if "config" in rec and "sim_metrics" in rec:
        flat: dict = {
            "trial":     rec.get("trial", 0),
            "timestamp": rec.get("timestamp", 0.0),
            "reward":    rec.get("reward", 0.0),
            "elapsed_s": rec.get("elapsed_s", 0.0),
        }
        flat.update(rec["config"])
        flat.update(rec["sim_metrics"])
        flat.update(rec.get("proxy_metrics", {}))
    else:
        flat = dict(rec)
        lanes = flat.get("mac_lanes", 8)
        flat.setdefault("accumulator_width",  32)
        flat.setdefault("clock_period_ns",    10)
        flat.setdefault("input_buffer_bytes",  1024)
        flat.setdefault("weight_buffer_bytes", 1024)
        flat.setdefault("area_proxy",   lanes / 8.0)
        flat.setdefault("power_proxy",  lanes / 8.0)
        flat.setdefault("timing_slack_ns", 10.0 - 2.0 - 0.12 * lanes)
        flat.setdefault("timing_violation",  flat["timing_slack_ns"] < 0)
        flat.setdefault("acc_overflow", False)
    flat["efficiency"] = flat.get("speedup", 1.0) / max(flat.get("mac_lanes", 1), 1)
    return flat


def load_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(RESULTS_FILE) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(_flatten(json.loads(s)))
            except (json.JSONDecodeError, KeyError):
                pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Helper drawing primitives ─────────────────────────────────────────────────

def _rect(shapes, x0, y0, x1, y1, fill, stroke="#888", lw=1.5):
    shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                       fillcolor=fill, line=dict(color=stroke, width=lw)))


def _line(shapes, x0, y0, x1, y1, color="#2a4466", lw=1.0, dash="dot"):
    shapes.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                       line=dict(color=color, width=lw, dash=dash)))


def _ann(annotations, x, y, text, color="#b0c8e0", size=9,
         anchor="center", family="monospace"):
    annotations.append(dict(
        x=x, y=y, text=text,
        font=dict(color=color, size=size, family=family),
        showarrow=False, xanchor=anchor, yanchor="middle",
    ))


def _sidebar_bar(shapes, annotations, bx0, bx1, y_base, height,
                 fill_frac, color, top_label, bot_label):
    """Vertical progress bar drawn outside the die (area / timing indicators)."""
    _rect(shapes, bx0, y_base, bx1, y_base + height, "rgba(18,22,34,0.9)", "#1e2840", 1)
    fh = height * max(0.0, min(fill_frac, 1.0))
    _rect(shapes, bx0, y_base, bx1, y_base + fh, color, color, 0)
    cx = (bx0 + bx1) / 2
    _ann(annotations, cx, y_base + height + 0.28, top_label, "#506070", 8)
    _ann(annotations, cx, y_base - 0.30, bot_label, color, 9)


# ── Chip Floorplan ────────────────────────────────────────────────────────────

def make_chip_floorplan(row: pd.Series) -> go.Figure:
    """
    Proportional block-level chip floorplan.

    Block sizes scale with the config so every combination looks visually distinct:
      mac_lanes  1→16 :  MAC array grows from 1 cell → 8×2 grid  (width + height)
      buffer_bytes    :  SRAM blocks grow taller (log2 scale)
      accumulator_w   :  accumulator bar thickness; turns red if int16 overflow
      timing_violation:  die border turns red, timing bar empties
    """
    lanes   = int(row.get("mac_lanes",           8))
    acc_w   = int(row.get("accumulator_width",   32))
    in_buf  = int(row.get("input_buffer_bytes",  1024))
    wt_buf  = int(row.get("weight_buffer_bytes", 1024))
    clk_ns  = float(row.get("clock_period_ns",   10))
    area    = float(row.get("area_proxy",         1.0))
    slack   = float(row.get("timing_slack_ns",    5.0))
    ovflw   = bool(row.get("acc_overflow",        False))
    t_viol  = bool(row.get("timing_violation",    False))
    speedup = float(row.get("speedup",            1.0))
    trial   = int(row.get("trial",                0))
    reward  = float(row.get("reward",             0.0))

    W, H  = 16.0, 12.0
    PAD   = 0.44
    GAP   = 0.22

    # ── Block sizing (all scale with config) ──────────────────────────────────

    # MAC array: columns first (max 8 wide), rows determined by lane count
    mac_cols = min(lanes, 8)
    mac_rows = math.ceil(lanes / mac_cols)
    cell_w, cell_h = 0.60, 0.46
    gap_x,  gap_y  = 0.13, 0.12
    mac_w = mac_cols * cell_w + (mac_cols - 1) * gap_x + 0.28
    mac_h = mac_rows * cell_h + (mac_rows - 1) * gap_y + 0.26

    # SRAM buffers: height scales log2 with byte count
    def buf_height(b: int) -> float:
        return max(1.0, min(2.5, 0.9 + 1.6 * (math.log2(b) - 8) / 4))  # log2(256)=8, log2(4096)=12

    in_h  = buf_height(in_buf)
    wt_h  = buf_height(wt_buf)
    buf_w = max(1.55, min(2.1, 0.85 + 1.25 * (math.log2(max(in_buf, wt_buf)) - 8) / 4))

    # Accumulator: thickness scales with acc_width
    acc_h = 0.36 + 0.34 * (acc_w / 32)   # 0.36 (16b) → 0.70 (32b)

    # Fixed blocks
    out_h  = 0.70
    ctrl_h = 0.55

    # ── Die dimensions (adapt to content) ────────────────────────────────────
    min_inner_w = buf_w + 0.4 + mac_w + 0.4 + buf_w
    die_w = min(max(min_inner_w + 2 * PAD, 8.5), W - 2.2)
    die_h = min(ctrl_h + GAP + out_h + GAP + acc_h + GAP +
                mac_h + GAP + max(in_h, wt_h) + 2 * PAD + 0.35,
                H - 2.0)

    die_x0 = (W - die_w) / 2
    die_y0 = (H - die_h) / 2
    die_x1 = die_x0 + die_w
    die_y1 = die_y0 + die_h
    dcx    = (die_x0 + die_x1) / 2   # die centre-x
    fx0    = die_x0 + PAD             # full-width block left edge
    fx1    = die_x1 - PAD             # full-width block right edge

    # ── Y layout (bottom → top) ───────────────────────────────────────────────
    y      = die_y0 + PAD
    ctrl_y0, ctrl_y1 = y, y + ctrl_h;  y = ctrl_y1 + GAP
    out_y0, out_y1   = y, y + out_h;   y = out_y1  + GAP
    acc_y0, acc_y1   = y, y + acc_h;   y = acc_y1  + GAP
    mac_y0, mac_y1   = y, y + mac_h;   y = mac_y1  + GAP
    buf_y0 = y

    # ── X layout: buffers + MAC array centred ─────────────────────────────────
    group_w = buf_w + 0.4 + mac_w + 0.4 + buf_w
    gx0    = dcx - group_w / 2
    in_x0, in_x1   = gx0, gx0 + buf_w
    mac_x0          = in_x1 + 0.4
    mac_x1          = mac_x0 + mac_w
    wt_x0, wt_x1   = mac_x1 + 0.4, mac_x1 + 0.4 + buf_w

    # Output row (requant | output SRAM) centred
    rq_w   = 1.75
    os_w   = 2.6
    row_w  = rq_w + 0.28 + os_w
    rq_x0, rq_x1   = dcx - row_w / 2, dcx - row_w / 2 + rq_w
    os_x0, os_x1   = rq_x1 + 0.28, rq_x1 + 0.28 + os_w

    # ── Build shapes / annotations ────────────────────────────────────────────
    shapes: list      = []
    annotations: list = []

    phase = int(time.time() * 2.3) % max(lanes, 1)   # drives MAC-cell animation

    # Canvas background
    _rect(shapes, 0, 0, W, H, BG_DEEP, BG_DEEP, 0)

    # Die outline (red if timing violation)
    die_border = C_RED if t_viol else C_BLUE
    _rect(shapes, die_x0, die_y0, die_x1, die_y1, BG_DIE, die_border, 2.0)

    # Fiducial corner marks (like real chip drawings)
    cs = 0.21
    for (fx, fy, dx, dy) in [
        (die_x0, die_y0,  1,  1), (die_x1, die_y0, -1,  1),
        (die_x0, die_y1,  1, -1), (die_x1, die_y1, -1, -1),
    ]:
        _line(shapes, fx, fy, fx + dx * cs, fy, die_border, 2.0, "solid")
        _line(shapes, fx, fy, fx,           fy + dy * cs, die_border, 2.0, "solid")

    # ── Layer 1: Control / FSM ────────────────────────────────────────────────
    _rect(shapes, fx0, ctrl_y0, fx1, ctrl_y1, "rgba(30,32,60,0.85)", "#404488", 1.0)
    _ann(annotations, dcx, (ctrl_y0 + ctrl_y1) / 2,
         "▶  Control / FSM  ·  register decode  ·  cmd sequencer  ·  status flags",
         "#6677aa", 9)

    # ── Layer 2: Output stage ─────────────────────────────────────────────────
    # Requantisation unit (colour red if overflow risk)
    rq_fill  = "rgba(140, 20, 20, 0.80)" if ovflw else "rgba(110, 52,  8, 0.80)"
    rq_color = C_RED if ovflw else "#cc8833"
    _rect(shapes, rq_x0, out_y0, rq_x1, out_y1, rq_fill, rq_color)
    _ann(annotations, (rq_x0 + rq_x1) / 2, (out_y0 + out_y1) / 2,
         f"Requant<br>{acc_w}b → int8{'  ⚠' if ovflw else ''}",
         C_RED if ovflw else "#ffcc88", 9)

    # Output SRAM
    _rect(shapes, os_x0, out_y0, os_x1, out_y1, "rgba(0,85,42,0.80)", "#22bb66")
    _ann(annotations, (os_x0 + os_x1) / 2, (out_y0 + out_y1) / 2,
         "Output SRAM<br>[int8 results]", "#88ffbb", 9)

    _line(shapes, rq_x1, (out_y0+out_y1)/2, os_x0, (out_y0+out_y1)/2, "#cc8833", 1.5, "solid")

    # ── Layer 3: Accumulator bank ─────────────────────────────────────────────
    acc_fill   = "rgba(150,20,20,0.80)"    if ovflw else "rgba(80, 8,120,0.80)"
    acc_border = C_RED                      if ovflw else C_PURPLE
    _rect(shapes, fx0, acc_y0, fx1, acc_y1, acc_fill, acc_border)
    ovf_warn = "  ⚠ OVERFLOW — int16 too narrow for TinyVAD" if ovflw else ""
    _ann(annotations, dcx, (acc_y0 + acc_y1) / 2,
         f"Accumulator Bank  [ {lanes} × {acc_w}-bit int ]{ovf_warn}",
         C_RED if ovflw else "#dd88ff", 10)

    # Wire: accumulator → requant
    _line(shapes, (rq_x0+rq_x1)/2, acc_y0, (rq_x0+rq_x1)/2, out_y1, "#9933cc", 1.2, "solid")

    # ── Layer 4: MAC array ────────────────────────────────────────────────────
    # Array container
    _rect(shapes, mac_x0 - 0.08, mac_y0 - 0.08, mac_x1 + 0.08, mac_y1 + 0.08,
          "rgba(0,12,35,0.7)", "#1a3055", 1.0)

    cell_start_x = mac_x0 + 0.14
    cell_start_y = mac_y0 + 0.13

    for i in range(lanes):
        ci   = i % mac_cols
        ri   = i // mac_cols
        cx_c = cell_start_x + ci * (cell_w + gap_x)
        cy_c = cell_start_y + ri * (cell_h + gap_y)

        if i == phase:            # currently computing
            fill, border, lw = C_ORANGE, "#ffffff", 2.0
        elif i < phase:           # done this cycle
            fill, border, lw = "rgba(0,165,78,0.65)", C_GREEN, 1.2
        else:                     # waiting
            fill, border, lw = "rgba(12,38,82,0.75)", "#1e3a60", 1.0

        _rect(shapes, cx_c, cy_c, cx_c + cell_w, cy_c + cell_h, fill, border, lw)

        # Label cells (omit if very small)
        if cell_w >= 0.44 and lanes <= 16:
            _ann(annotations, cx_c + cell_w / 2, cy_c + cell_h / 2,
                 f"M{i}", "white", 8)

    # MAC array label (above the array, inside the die)
    _ann(annotations, dcx, mac_y1 + 0.14,
         f"int8 MAC Array  ·  {lanes} lane{'s' if lanes > 1 else ''}  ·  "
         f"each lane: 8b × 8b → {acc_w}b accumulator",
         C_CYAN, 10)

    # Wire: MAC → accumulator
    _line(shapes, dcx, mac_y0, dcx, acc_y1, "#2255aa", 1.2, "solid")

    # ── Layer 5: SRAM buffers (flank the MAC array) ───────────────────────────
    _rect(shapes, in_x0, buf_y0, in_x1, buf_y0 + in_h, "rgba(0,52,110,0.80)", C_BLUE)
    _ann(annotations, (in_x0 + in_x1) / 2, buf_y0 + in_h / 2,
         f"Input<br>SRAM<br>{in_buf} B", "#88ccff", 9)

    _rect(shapes, wt_x0, buf_y0, wt_x1, buf_y0 + wt_h, "rgba(0,52,110,0.80)", C_BLUE)
    _ann(annotations, (wt_x0 + wt_x1) / 2, buf_y0 + wt_h / 2,
         f"Weight<br>SRAM<br>{wt_buf} B", "#88ccff", 9)

    # Wires: buffers → MAC
    _line(shapes, (in_x0+in_x1)/2, buf_y0, mac_x0 + mac_w*0.28, mac_y1, "#2266aa", 1.2, "solid")
    _line(shapes, (wt_x0+wt_x1)/2, buf_y0, mac_x0 + mac_w*0.72, mac_y1, "#2266aa", 1.2, "solid")

    # Memory bus strip (top of die, above buffers)
    bus_y0 = buf_y0 + max(in_h, wt_h) + 0.12
    if bus_y0 + 0.35 < die_y1 - 0.08:
        _rect(shapes, fx0, bus_y0, fx1, die_y1 - 0.08, "rgba(0,25,15,0.45)", "#1a5533", 1.0)
        _ann(annotations, dcx, (bus_y0 + die_y1 - 0.08) / 2,
             "← Memory Bus  (PicoRV32 addr / data / wstrb) →", "#2a5040", 8)

    # ── Side bars (area + timing indicators) ──────────────────────────────────
    bar_h = die_y1 - die_y0

    # Area bar  (right of die)
    area_color = (C_RED if area > 1.5 else C_ORANGE if area > 1.0 else C_GREEN)
    _sidebar_bar(shapes, annotations,
                 die_x1 + 0.14, die_x1 + 0.40, die_y0, bar_h,
                 min(area / 2.0, 1.0), area_color,
                 "AREA", f"{area:.2f}×")

    # Timing slack bar  (left of die)
    slack_frac  = max(0.0, min(slack / 15.0, 1.0))
    slack_color = (C_RED if slack < 0 else C_ORANGE if slack < 2 else C_GREEN)
    _sidebar_bar(shapes, annotations,
                 die_x0 - 0.40, die_x0 - 0.14, die_y0, bar_h,
                 slack_frac, slack_color,
                 "SLACK", f"{slack:+.1f}ns")

    # ── Stats overlay (top-right corner of die) ───────────────────────────────
    timing_str   = "⚠ TIMING VIOLATION" if t_viol else f"slack {slack:+.1f}ns  ✓"
    timing_color = C_RED if t_viol else C_GREEN
    stats = [
        (f"trial  {trial}",                           "#8899bb"),
        (f"lanes  {lanes}",                           C_CYAN),
        (f"acc    {acc_w}b",                          C_PURPLE),
        (f"clk    {clk_ns}ns  ({1000/clk_ns:.0f}MHz)", "#88aacc"),
        (f"speed  {speedup:.1f}×",                    C_GREEN),
        (f"area   {area:.3f}",                        area_color),
        (timing_str,                                   timing_color),
        (f"reward {reward:+.3f}",                     C_GOLD),
    ]
    for j, (txt, col) in enumerate(stats):
        _ann(annotations, die_x1 - 0.10, die_y1 - 0.38 - j * 0.37,
             txt, col, 9, "right")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=[0, W], visible=False, fixedrange=True),
        yaxis=dict(range=[0, H], visible=False, fixedrange=True, scaleanchor="x"),
        plot_bgcolor=BG_DEEP, paper_bgcolor=BG_PANEL,
        margin=dict(t=44, b=8, l=8, r=8), height=560,
        title=dict(
            text=(f"<b>Chip Floorplan</b>"
                  f"  ·  {lanes}-lane MAC"
                  f"  ·  {acc_w}b acc"
                  f"  ·  {1000/clk_ns:.0f} MHz"
                  f"  ·  area {area:.2f}×"
                  f"  ·  {'<span style=\"color:#ff3355\">⚠ TIMING VIOLATION</span>' if t_viol else 'timing OK'}"),
            font=dict(color=C_CYAN, size=13), x=0.5,
        ),
    )
    return fig


# ── PPA Evolution (4-panel live graphs) ───────────────────────────────────────

def make_ppa_evolution(df: pd.DataFrame) -> go.Figure:
    """
    2 × 2 subplot grid showing all four PPA dimensions vs trial number.
    Each new trial adds a data point; the gold dot marks the most recent.
    Timing-violation trials appear as red ✕ markers on the slack panel.
    """
    ds = df.sort_values("trial").reset_index(drop=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Performance  (speedup ×)", "Area  (proxy, 1.0 = baseline)",
                        "Power  (proxy, 1.0 = baseline)", "Timing Slack  (ns, ↑ better)"],
        vertical_spacing=0.20,
        horizontal_spacing=0.14,
    )

    def _trail(x, y, color, row, col):
        """Line + dots trail; gold marker on the latest point."""
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", showlegend=False,
            line=dict(color=color, width=2.0),
            marker=dict(size=5, color=color, line=dict(color="white", width=0.5)),
        ), row=row, col=col)
        if len(x):
            fig.add_trace(go.Scatter(
                x=[x.iloc[-1]], y=[y.iloc[-1]], mode="markers", showlegend=False,
                marker=dict(symbol="circle", size=13, color=C_GOLD,
                            line=dict(color="black", width=1.5)),
            ), row=row, col=col)

    # ── Speedup ──────────────────────────────────────────────────────────────
    _trail(ds["trial"], ds["speedup"], C_GREEN, 1, 1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#2a3a44",
                  annotation_text=" SW baseline",
                  annotation_font_color="#445566",
                  annotation_position="top right", row=1, col=1)

    # ── Area ─────────────────────────────────────────────────────────────────
    if "area_proxy" in ds.columns:
        _trail(ds["trial"], ds["area_proxy"], C_ORANGE, 1, 2)
        fig.add_hline(y=1.0, line_dash="dot", line_color="#2a3a44",
                      annotation_text=" baseline",
                      annotation_font_color="#445566",
                      annotation_position="top right", row=1, col=2)

    # ── Power ─────────────────────────────────────────────────────────────────
    if "power_proxy" in ds.columns:
        _trail(ds["trial"], ds["power_proxy"], C_PINK, 2, 1)
        fig.add_hline(y=1.0, line_dash="dot", line_color="#2a3a44",
                      annotation_text=" baseline",
                      annotation_font_color="#445566",
                      annotation_position="top right", row=2, col=1)

    # ── Timing slack ──────────────────────────────────────────────────────────
    if "timing_slack_ns" in ds.columns:
        slack = ds["timing_slack_ns"]
        dot_colors = [C_RED if v < 0 else (C_ORANGE if v < 2 else C_GREEN) for v in slack]
        fig.add_trace(go.Scatter(
            x=ds["trial"], y=slack, mode="lines+markers", showlegend=False,
            line=dict(color=C_CYAN, width=2.0),
            marker=dict(size=6, color=dot_colors, line=dict(color="white", width=0.5)),
        ), row=2, col=2)
        # Latest gold dot
        fig.add_trace(go.Scatter(
            x=[ds["trial"].iloc[-1]], y=[slack.iloc[-1]],
            mode="markers", showlegend=False,
            marker=dict(symbol="circle", size=13, color=C_GOLD,
                        line=dict(color="black", width=1.5)),
        ), row=2, col=2)
        # Zero-line = violation boundary
        fig.add_hline(y=0, line_dash="dash", line_color=C_RED,
                      annotation_text=" violation",
                      annotation_font_color=C_RED,
                      annotation_position="top right", row=2, col=2)
        # Red X on every violation trial
        if "timing_violation" in ds.columns:
            viols = ds[ds["timing_violation"] == True]
            if len(viols):
                fig.add_trace(go.Scatter(
                    x=viols["trial"], y=viols["timing_slack_ns"],
                    mode="markers", showlegend=False,
                    marker=dict(symbol="x", size=14, color=C_RED,
                                line=dict(color=C_RED, width=2)),
                ), row=2, col=2)

    # ── Styling ───────────────────────────────────────────────────────────────
    axis_style = dict(gridcolor=GRID_CLR, color="#6a8aaa", zerolinecolor=GRID_CLR,
                      tickfont=dict(color="#6a8aaa"))
    for i in range(1, 5):
        fig.update_layout(**{
            f"xaxis{'' if i==1 else i}": dict(title="Trial #", **axis_style),
            f"yaxis{'' if i==1 else i}": axis_style,
        })

    for ann in fig.layout.annotations:
        if ann.text in ["Performance  (speedup ×)", "Area  (proxy, 1.0 = baseline)",
                        "Power  (proxy, 1.0 = baseline)", "Timing Slack  (ns, ↑ better)"]:
            ann.font.color = "#88aacc"
            ann.font.size  = 11

    fig.update_layout(
        plot_bgcolor=BG_DEEP, paper_bgcolor=BG_PANEL,
        margin=dict(t=55, b=30, l=50, r=20), height=560,
        title=dict(text="<b>PPA Evolution</b>  ·  live metrics per trial",
                   font=dict(color=C_CYAN, size=13), x=0.5),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Trial History Strip ───────────────────────────────────────────────────────

def make_trial_strip(df: pd.DataFrame) -> go.Figure:
    """
    One coloured cell per trial.  Green = high reward, red = low / violation.
    Each bar is labelled with its mac_lanes value.
    The gold star marks the best trial so far.
    This gives the "timeline of exploration" sense at a glance.
    """
    ds = df.sort_values("trial")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=ds["trial"],
        y=[1.0] * len(ds),
        marker=dict(
            color=ds["reward"],
            colorscale=[
                [0.0, "#440010"], [0.3, "#882200"],
                [0.5, "#886600"], [0.7, "#227733"],
                [1.0, "#00cc66"],
            ],
            cmin=ds["reward"].min(),
            cmax=ds["reward"].max(),
            colorbar=dict(
                title=dict(text="Reward", font=dict(color="#6a8aaa", size=10)),
                tickfont=dict(color="#6a8aaa", size=9),
                thickness=10, len=0.85, x=1.01,
            ),
            line=dict(color="rgba(0,0,0,0.25)", width=0.5),
        ),
        text=[f"L={int(l)}" for l in ds["mac_lanes"]],
        textfont=dict(color="white", size=8),
        textposition="inside",
        hovertemplate=(
            "Trial %{x}<br>"
            "reward = %{marker.color:.3f}<br>"
            "mac_lanes = %{text}<extra></extra>"
        ),
        width=0.85,
    ))

    # Gold star on best trial
    best_idx = ds["reward"].idxmax()
    best_row = ds.loc[best_idx]
    fig.add_trace(go.Scatter(
        x=[best_row["trial"]], y=[1.12],
        mode="markers", showlegend=False,
        marker=dict(symbol="star", size=16, color=C_GOLD,
                    line=dict(color="black", width=1)),
        hovertemplate=f"Best  trial={int(best_row['trial'])}  reward={best_row['reward']:.3f}<extra></extra>",
    ))

    # Vertical red lines at timing violations
    if "timing_violation" in ds.columns:
        for _, vrow in ds[ds["timing_violation"] == True].iterrows():
            fig.add_vline(x=vrow["trial"], line_color=C_RED,
                          line_width=1, line_dash="dot",
                          annotation_text="⚠", annotation_font_color=C_RED,
                          annotation_position="top")

    fig.update_layout(
        plot_bgcolor=BG_DEEP, paper_bgcolor=BG_PANEL,
        margin=dict(t=44, b=22, l=50, r=70), height=130,
        title=dict(text="<b>Trial History</b>  ·  each bar = one evaluated config  ·  green = high reward  ·  L = mac_lanes  ·  ⭐ = best",
                   font=dict(color=C_CYAN, size=11), x=0.5),
        xaxis=dict(title="Trial #", color="#6a8aaa", gridcolor=GRID_CLR,
                   tickmode="auto", nticks=min(len(ds), 25)),
        yaxis=dict(visible=False, range=[0, 1.3]),
        bargap=0.08, showlegend=False, font=dict(color="#99bbcc"),
    )
    return fig


# ── Pareto frontier ───────────────────────────────────────────────────────────

def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only non-dominated rows.

    Objectives: maximise speedup (↑), minimise area_proxy (↓).
    Row A dominates row B iff A.speedup ≥ B.speedup AND A.area ≤ B.area
    with at least one strict inequality.  Only correct/non-overflowing configs
    are eligible.
    """
    if df.empty:
        return df

    area_col  = "area_proxy"  if "area_proxy"  in df.columns else None
    overflow_col = "acc_overflow" if "acc_overflow" in df.columns else None

    # Only correct configs can be Pareto-optimal
    cands = df[df["accuracy"] >= 1.0].copy() if "accuracy" in df.columns else df.copy()
    if overflow_col:
        cands = cands[cands[overflow_col] != True]
    if cands.empty:
        return cands

    if area_col is None:
        # No area column — return configs with highest speedup only
        return cands[cands["speedup"] == cands["speedup"].max()]

    front_idx = []
    speedups = cands["speedup"].values
    areas    = cands[area_col].values
    indices  = cands.index.tolist()

    for i, idx in enumerate(indices):
        dominated = False
        for j in range(len(indices)):
            if i == j:
                continue
            # j dominates i?
            if (speedups[j] >= speedups[i] and areas[j] <= areas[i] and
                    (speedups[j] > speedups[i] or areas[j] < areas[i])):
                dominated = True
                break
        if not dominated:
            front_idx.append(idx)

    return cands.loc[front_idx].sort_values("speedup")


def make_pareto(df: pd.DataFrame) -> go.Figure:
    """
    Scatter of all trials with the true Pareto-optimal front highlighted.

    The front is computed by non-domination filtering on (speedup ↑, area ↓).
    Dominated trials are shown as faded grey circles; frontier configs as
    bright circles connected by a step line.  The SW baseline is a reference
    dashed line.
    """
    front = _pareto_front(df)
    dominated = df.drop(index=front.index, errors="ignore")

    def _hover(row_df):
        return [
            f"Trial {t}<br>lanes={ml}  acc={aw}b  clk={clk}ns<br>"
            f"{cyc:,} cycles  {sp:.1f}× speedup<br>"
            f"area={ar:.2f}  slack={sl:+.1f}ns  reward={rw:.3f}"
            for t, ml, aw, clk, cyc, sp, ar, sl, rw in zip(
                row_df["trial"], row_df["mac_lanes"],
                row_df.get("accumulator_width",  pd.Series([32]  * len(row_df))),
                row_df.get("clock_period_ns",    pd.Series([10]  * len(row_df))),
                row_df["avg_cycles"], row_df["speedup"],
                row_df.get("area_proxy",         pd.Series([1.0] * len(row_df))),
                row_df.get("timing_slack_ns",    pd.Series([5.0] * len(row_df))),
                row_df["reward"],
            )
        ]

    fig = go.Figure()

    # Dominated configs — faded background
    if not dominated.empty:
        fig.add_trace(go.Scatter(
            x=dominated["mac_lanes"], y=dominated["avg_cycles"],
            mode="markers", name="Dominated",
            marker=dict(size=8, color="rgba(80,100,120,0.35)",
                        line=dict(color="rgba(120,140,160,0.3)", width=0.5)),
            text=_hover(dominated),
            hovertemplate="%{text}<extra></extra>",
        ))

    # Pareto-optimal configs — bright + step line
    if not front.empty:
        fig.add_trace(go.Scatter(
            x=front["mac_lanes"], y=front["avg_cycles"],
            mode="markers+lines", name="Pareto front",
            line=dict(color=C_CYAN, width=1.5, dash="dot", shape="hv"),
            marker=dict(
                size=front["speedup"] * 2.2,
                color=front["reward"], colorscale="RdYlGn",
                colorbar=dict(title=dict(text="Reward", font=dict(color="#6a8aaa")),
                              tickfont=dict(color="#6a8aaa")),
                line=dict(color="white", width=1.2),
            ),
            text=_hover(front),
            hovertemplate="%{text}<extra></extra>",
        ))
        # Gold star on highest-reward Pareto config
        best = front.loc[front["reward"].idxmax()]
        fig.add_trace(go.Scatter(
            x=[best["mac_lanes"]], y=[best["avg_cycles"]], mode="markers",
            marker=dict(symbol="star", size=22, color=C_GOLD,
                        line=dict(color="black", width=1)),
            name="Best on front", showlegend=False,
        ))

    fig.add_hline(y=SW_BASELINE, line_dash="dash", line_color="#334455",
                  annotation_text=" SW baseline (no accel)",
                  annotation_font_color="#556677",
                  annotation_position="top right")

    n_front = len(front)
    n_all   = len(df)
    fig.update_layout(
        plot_bgcolor=BG_DEEP, paper_bgcolor=BG_PANEL,
        margin=dict(t=44, b=30, l=50, r=10), height=360,
        title=dict(
            text=(f"<b>Pareto Frontier</b>  ·  speedup ↑  vs  area ↓"
                  f"  ·  {n_front} / {n_all} configs non-dominated"),
            font=dict(color=C_CYAN, size=13), x=0.5,
        ),
        xaxis=dict(title="MAC lanes  (area proxy →)", color="#6a8aaa",
                   gridcolor=GRID_CLR, tickvals=[1, 2, 4, 8, 16]),
        yaxis=dict(title="Avg cycles / inference  (↓ better)",
                   color="#6a8aaa", gridcolor=GRID_CLR),
        legend=dict(font=dict(color="#99bbcc"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Parallel coordinates ──────────────────────────────────────────────────────

def make_parallel_coords(df: pd.DataFrame) -> go.Figure:
    """All design dimensions on one plot — drag axes to filter configs."""

    def _dim(col, label, tick_vals=None, tick_text=None):
        if col not in df.columns:
            return None
        # Only use properties valid for go.parcoords.Dimension:
        # label, values, tickvals, ticktext, tickformat, range, visible, constraintrange
        # NOTE: tickfont and labelstyle are NOT valid — they live on go.Parcoords, not Dimension
        d = dict(label=label, values=df[col].tolist())
        if tick_vals:
            d["tickvals"] = tick_vals
            d["ticktext"] = tick_text or [str(v) for v in tick_vals]
        return d

    dfc = df.copy()
    if "dataflow" in dfc.columns:
        dfc["dataflow_num"] = dfc["dataflow"].map(
            {"output_stationary": 0, "weight_stationary": 1}).fillna(0)

    dims = [
        _dim("mac_lanes",          "MAC Lanes",  [1,2,4,8,16]),
        _dim("accumulator_width",  "Acc Width b",[16,24,32]),
        _dim("clock_period_ns",    "Clock ns",   [5,10,20]),
        _dim("input_buffer_bytes", "In Buf B",   [256,512,1024,2048,4096]),
        _dim("weight_buffer_bytes","Wt Buf B",   [256,512,1024,2048,4096]),
        _dim("speedup",            "Speedup ×"),
        _dim("area_proxy",         "Area"),
        _dim("power_proxy",        "Power"),
        _dim("timing_slack_ns",    "Slack ns"),
        _dim("reward",             "Reward"),
    ]
    dims = [d for d in dims if d is not None]

    fig = go.Figure(go.Parcoords(
        line=dict(color=dfc["reward"], colorscale="RdYlGn",
                  showscale=True,
                  colorbar=dict(title=dict(text="Reward", font=dict(color="#6a8aaa")),
                                tickfont=dict(color="#6a8aaa"), thickness=10)),
        dimensions=dims,
        labelfont=dict(color="#aaccdd", size=10),
    ))
    fig.update_layout(
        plot_bgcolor=BG_DEEP, paper_bgcolor=BG_PANEL,
        margin=dict(t=50, b=30, l=60, r=60), height=320,
        title=dict(text="<b>Parallel Coordinates</b>  ·  drag axes to filter  ·  each line = one trial",
                   font=dict(color=C_CYAN, size=13), x=0.5),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Main render loop ──────────────────────────────────────────────────────────

st.markdown(
    "<h2 style='color:#00d4ff;margin-bottom:2px'>⚡ TinyMAC Design-Space Explorer</h2>"
    f"<p style='color:#506070;margin-top:0'>"
    f"Stage 3 SW baseline: <b style='color:#7a9aaa'>{SW_BASELINE:,} cycles/inference</b>"
    f" &nbsp;·&nbsp; floorplan blocks scale with each config"
    f" &nbsp;·&nbsp; auto-refresh every {REFRESH_SEC}s</p>",
    unsafe_allow_html=True,
)

df = load_results()

# ── Empty-state waiting screen ────────────────────────────────────────────────
if df.empty:
    st.markdown(
        "<div style='background:#10141f;border:1px solid #1e2840;border-radius:10px;"
        "padding:28px 36px;margin-top:20px'>"
        "<p style='color:#6a8aaa;font-size:1.05rem;margin-bottom:12px'>"
        "No trials yet — start the optimizer in a WSL terminal:</p>"
        "<pre style='color:#00d4ff;background:#080a12;padding:14px 18px;"
        "border-radius:6px;font-size:0.95rem'>"
        "cd ~/voiceAI\n"
        "python3 optimizer/run_optimizer.py                   # evo agent, 30 trials\n"
        "python3 optimizer/run_optimizer.py --agent ucb       # UCB1 bandit\n"
        "python3 optimizer/run_optimizer.py --agent bayesian  # Optuna TPE\n"
        "python3 optimizer/run_optimizer.py --agent random    # random baseline"
        "</pre>"
        "<p style='color:#405060;margin-top:12px;font-size:0.85rem'>"
        "SSH tunnel: <code style='color:#7090a0'>ssh -L 8501:localhost:8501 user@vm</code>"
        "</p></div>",
        unsafe_allow_html=True,
    )
    time.sleep(REFRESH_SEC)
    st.rerun()

# ── KPI row ───────────────────────────────────────────────────────────────────
best    = df.loc[df["reward"].idxmax()]
n_viols = int(df.get("timing_violation", pd.Series([False]*len(df))).sum())
n_ovflw = int(df.get("acc_overflow",     pd.Series([False]*len(df))).sum())

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Trials",        len(df))
c2.metric("Best lanes",    int(best["mac_lanes"]))
c3.metric("Best speedup",  f"{best['speedup']:.1f}×")
c4.metric("Best cycles",   f"{int(best['avg_cycles']):,}")
c5.metric("Best reward",   f"{best['reward']:.3f}")
c6.metric("Timing viols",  n_viols,  delta=f"{'⚠ ' if n_viols else '✓ '}{'violations' if n_viols else 'none'}", delta_color="inverse")
c7.metric("Acc overflows", n_ovflw,  delta=f"{'⚠ ' if n_ovflw else '✓ '}{'configs' if n_ovflw else 'none'}", delta_color="inverse")

st.markdown("<hr style='margin:8px 0 12px 0'>", unsafe_allow_html=True)

# ── Row 1: floorplan + PPA evolution ─────────────────────────────────────────
col_fp, col_ppa = st.columns([1, 1])
latest = df.iloc[-1]

with col_fp:
    st.plotly_chart(make_chip_floorplan(latest), use_container_width=True)

with col_ppa:
    st.plotly_chart(make_ppa_evolution(df), use_container_width=True)

# ── Row 2: trial history strip ────────────────────────────────────────────────
st.plotly_chart(make_trial_strip(df), use_container_width=True)

st.markdown("<hr style='margin:4px 0 10px 0'>", unsafe_allow_html=True)

# ── Row 3: Pareto + parallel coords ──────────────────────────────────────────
col_par, col_pc = st.columns([1, 1])

with col_par:
    st.plotly_chart(make_pareto(df), use_container_width=True)

with col_pc:
    if len(df) >= 3:
        st.plotly_chart(make_parallel_coords(df), use_container_width=True)
    else:
        st.info("Parallel coordinates appears after 3+ trials.")

# ── Row 4: trial table ────────────────────────────────────────────────────────
show = [c for c in [
    "trial", "mac_lanes", "accumulator_width", "clock_period_ns",
    "input_buffer_bytes", "weight_buffer_bytes",
    "avg_cycles", "speedup", "efficiency",
    "area_proxy", "power_proxy", "timing_slack_ns",
    "acc_overflow", "accuracy", "reward", "elapsed_s",
] if c in df.columns]

with st.expander("All trials", expanded=False):
    disp = df[show].sort_values("reward", ascending=False).reset_index(drop=True)
    fmts = {
        "avg_cycles":      lambda x: f"{int(x):,}",
        "speedup":         lambda x: f"{x:.1f}×",
        "efficiency":      lambda x: f"{x:.2f}×",
        "area_proxy":      lambda x: f"{x:.3f}",
        "power_proxy":     lambda x: f"{x:.3f}",
        "timing_slack_ns": lambda x: f"{x:+.2f}",
        "accuracy":        lambda x: f"{float(x):.0%}",
        "reward":          lambda x: f"{x:.3f}",
        "elapsed_s":       lambda x: f"{float(x):.1f}s",
    }
    for col, fn in fmts.items():
        if col in disp.columns:
            disp[col] = disp[col].apply(fn)
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<p style='color:#2a3a4a;font-size:0.78rem;text-align:right;margin-top:8px'>"
    f"auto-refresh every {REFRESH_SEC}s  ·  {time.strftime('%H:%M:%S')}  ·  {RESULTS_FILE}"
    "</p>",
    unsafe_allow_html=True,
)
time.sleep(REFRESH_SEC)
st.rerun()

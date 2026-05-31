"""dashboard.py — live Streamlit dashboard for the TinyMAC design-space explorer.

Layout:
  Row 0  : KPI metrics bar
  Row 1  : animated MAC-array schematic  |  reward convergence curve
  Row 2  : Pareto scatter (cycles vs area) | speedup bar chart
  Row 3  : parallel coordinates (all dimensions at once)
  Row 4  : full trial table (collapsible)

Launch:
    streamlit run optimizer/dashboard.py

Access from a company VM without inbound ports (SSH tunnel):
    ssh -L 8501:localhost:8501 user@vm  →  open http://localhost:8501 locally

Backward compatible: handles both old flat records and new nested records
written by run_optimizer.py / env.py.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

RESULTS_FILE = Path(__file__).parent / "results.jsonl"
SW_BASELINE  = 175_324    # Stage 3 pure-software avg cycles/inference
REFRESH_SEC  = 2

DARK_BG       = "rgba(15,17,26,1)"
PANEL_BG      = "rgba(22,26,40,1)"
GRID_COLOR    = "rgba(255,255,255,0.06)"
ACCENT_CYAN   = "#00d4ff"
ACCENT_ORANGE = "#ff6b35"
ACCENT_GREEN  = "#00ff88"
ACCENT_GOLD   = "#ffd700"
ACCENT_RED    = "#ff4455"

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TinyMAC Design Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp { background: #0f111a; color: #e0e6f0; }
  section[data-testid="stSidebar"] { background: #0f111a; }
  div[data-testid="metric-container"] {
      background: #161a28; border: 1px solid #2a3050;
      border-radius: 8px; padding: 12px 16px;
  }
  div[data-testid="metric-container"] label { color: #7090b0 !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
      color: #00d4ff !important; font-size: 1.6rem !important;
  }
  hr { border-color: #2a3050; }
</style>
""", unsafe_allow_html=True)


# ── Data loading & normalization ───────────────────────────────────────────────

def _flatten(rec: dict) -> dict:
    """Normalize both old (flat) and new (nested) result record formats."""
    if "config" in rec and "sim_metrics" in rec:
        # New format written by env.py / run_optimizer.py
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
        # Old flat format written by search.py
        flat = dict(rec)
        lanes = flat.get("mac_lanes", 8)
        flat.setdefault("accumulator_width", 32)
        flat.setdefault("clock_period_ns", 10)
        flat.setdefault("input_buffer_bytes", 1024)
        flat.setdefault("weight_buffer_bytes", 1024)
        flat.setdefault("dataflow", "output_stationary")
        flat.setdefault("area_proxy",  (lanes / 8) * 1.0)
        flat.setdefault("power_proxy", (lanes / 8) * 1.0)
        flat.setdefault("timing_slack_ns", 10.0 - 2.0 - 0.12 * lanes)
        flat.setdefault("timing_violation", flat["timing_slack_ns"] < 0)
        flat.setdefault("acc_overflow", False)

    flat["efficiency"] = flat.get("speedup", 1.0) / max(flat.get("mac_lanes", 1), 1)
    return flat


def load_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(RESULTS_FILE) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(_flatten(json.loads(s)))
            except (json.JSONDecodeError, KeyError):
                pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Accelerator schematic ──────────────────────────────────────────────────────

def make_accel_diagram(row: pd.Series) -> go.Figure:
    mac_lanes  = int(row.get("mac_lanes", 8))
    acc_w      = int(row.get("accumulator_width", 32))
    avg_cycles = int(row.get("avg_cycles", 0))
    trial      = int(row.get("trial", 0))
    area       = float(row.get("area_proxy", 1.0))
    t_viol     = bool(row.get("timing_violation", False))

    W, H = 12.0, 9.0
    cols = min(mac_lanes, 8)
    rows_grid = math.ceil(mac_lanes / cols)
    mw, mh = 0.85, 0.65
    gx, gy = 1.15, 0.90
    arr_w  = cols * gx - (gx - mw)
    arr_h  = rows_grid * gy - (gy - mh)
    arr_x0 = (W - arr_w) / 2
    arr_y0 = 3.6

    phase = int(time.time() * 2.5) % mac_lanes
    shapes, annotations, sx, sy = [], [], [], []

    # Die border (red tint if timing violation)
    border_col = ACCENT_RED if t_viol else ACCENT_CYAN
    shapes.append(dict(type="rect", x0=0.3, y0=0.3, x1=W-0.3, y1=H-0.3,
                       fillcolor=PANEL_BG, line=dict(color=border_col, width=2),
                       layer="below"))

    def box(x0, y0, x1, y1, fill, stroke, lw=1.5):
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           fillcolor=fill, line=dict(color=stroke, width=lw)))

    def arrow(x0, y0, x1, y1, color="#336688"):
        shapes.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color=color, width=1.5, dash="dot")))

    # Buffers
    box(0.7, 0.7, 2.8, 2.2, "rgba(0,80,140,0.6)", "#3399cc")
    annotations.append(dict(x=1.75, y=1.45, text="<b>Input</b><br>Buffer<br>[49×40]",
                             font=dict(color="#88ccff", size=11), showarrow=False, align="center"))
    box(W-2.8, 0.7, W-0.7, 2.2, "rgba(0,80,140,0.6)", "#3399cc")
    annotations.append(dict(x=W-1.75, y=1.45, text="<b>Weight</b><br>Buffer<br>[int8]",
                             font=dict(color="#88ccff", size=11), showarrow=False, align="center"))
    box(W/2-1.1, 0.7, W/2+1.1, 2.2, "rgba(60,40,100,0.6)", "#9966cc")
    annotations.append(dict(x=W/2, y=1.45, text=f"<b>Bias</b><br>Q-mult<br>{acc_w}b acc",
                             font=dict(color="#cc99ff", size=11), showarrow=False, align="center"))

    arrow(1.75, 2.2, arr_x0 + arr_w*0.25, arr_y0)
    arrow(W-1.75, 2.2, arr_x0 + arr_w*0.75, arr_y0)
    arrow(W/2, 2.2, arr_x0 + arr_w*0.5, arr_y0)

    # MAC lane cells
    for i in range(mac_lanes):
        c, r = i % cols, i // cols
        x0, y0 = arr_x0 + c*gx, arr_y0 + r*gy
        x1, y1 = x0 + mw, y0 + mh
        if i == phase:
            fill, border, lw = ACCENT_ORANGE, "#ffffff", 2.0
        elif i < phase:
            fill, border, lw = "rgba(0,180,90,0.55)", ACCENT_GREEN, 1.0
        else:
            fill, border, lw = "rgba(30,50,90,0.7)", "#445577", 1.0
        box(x0, y0, x1, y1, fill, border, lw)
        sx.append((x0+x1)/2); sy.append((y0+y1)/2)

    # Output buffer
    out_x0 = W/2 - 1.4; out_x1 = W/2 + 1.4
    out_y0 = arr_y0 + arr_h + 0.55; out_y1 = out_y0 + 1.0
    box(out_x0, out_y0, out_x1, out_y1, "rgba(0,120,60,0.55)", ACCENT_GREEN)
    annotations.append(dict(x=W/2, y=(out_y0+out_y1)/2,
                             text="<b>Output Buffer</b>  [int8]",
                             font=dict(color=ACCENT_GREEN, size=11), showarrow=False))
    arrow(arr_x0+arr_w/2, arr_y0+arr_h, W/2, out_y0, ACCENT_GREEN)

    # Requant unit
    rq_x0, rq_x1 = W-2.7, W-0.7
    rq_y0, rq_y1 = arr_y0, arr_y0+0.9
    box(rq_x0, rq_y0, rq_x1, rq_y1, "rgba(120,60,0,0.5)", "#ffaa44")
    annotations.append(dict(x=(rq_x0+rq_x1)/2, y=(rq_y0+rq_y1)/2,
                             text=f"Requant<br>{acc_w}→int8",
                             font=dict(color="#ffcc88", size=10), showarrow=False))
    arrow(arr_x0+arr_w, arr_y0+arr_h/2, rq_x0, (rq_y0+rq_y1)/2, "#ffaa44")
    arrow(rq_x0+(rq_x1-rq_x0)/2, rq_y1, W/2+1.4, (out_y0+out_y1)/2, "#ffaa44")

    # Stats overlay
    viol_str = " ⚠ TIMING VIOL" if t_viol else ""
    stats = [
        f"mac_lanes : {mac_lanes}",
        f"acc width : {acc_w}b",
        f"area proxy: {area:.2f}×",
        f"avg cycles: {avg_cycles:,}",
        f"speedup   : {SW_BASELINE/max(avg_cycles,1):.1f}×",
        f"trial     : {trial}{viol_str}",
    ]
    for j, txt in enumerate(stats):
        annotations.append(dict(
            x=0.55, y=H - 0.50 - j*0.40,
            text=f"<span style='font-family:monospace;font-size:10px'>{txt}</span>",
            font=dict(color=ACCENT_RED if ("VIOL" in txt and t_viol) else "#99bbcc", size=10),
            showarrow=False, xanchor="left",
        ))

    fig = go.Figure()
    if sx:
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="text",
                                 text=[f"M{i}" for i in range(mac_lanes)],
                                 textfont=dict(color="white", size=9),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=[0,W], visible=False, fixedrange=True),
        yaxis=dict(range=[0,H], visible=False, fixedrange=True, scaleanchor="x"),
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        margin=dict(t=40, b=8, l=8, r=8), height=440,
        title=dict(
            text=(f"<b>TinyMAC Accelerator</b>  ·  "
                  f"{mac_lanes} lane{'s' if mac_lanes>1 else ''}  "
                  f"·  {acc_w}b acc  ·  Trial {trial}"),
            font=dict(color=ACCENT_CYAN, size=14), x=0.5,
        ),
    )
    return fig


# ── Reward convergence ─────────────────────────────────────────────────────────

def make_convergence(df: pd.DataFrame) -> go.Figure:
    ds = df.sort_values("trial")
    ds = ds.assign(best_so_far=ds["reward"].cummax())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds["trial"], y=ds["reward"],
                             mode="markers+lines", name="Trial reward",
                             line=dict(color="#4477ff", width=1.5, dash="dot"),
                             marker=dict(size=7, color="#4477ff",
                                         line=dict(color="white", width=1))))
    fig.add_trace(go.Scatter(x=ds["trial"], y=ds["best_so_far"], mode="lines",
                             name=f"Best {ds['best_so_far'].iloc[-1]:.3f}",
                             line=dict(color=ACCENT_GREEN, width=2.5)))
    best_row = ds.loc[ds["reward"].idxmax()]
    fig.add_trace(go.Scatter(x=[best_row["trial"]], y=[best_row["reward"]],
                             mode="markers",
                             marker=dict(symbol="star", size=18, color=ACCENT_GOLD,
                                         line=dict(color="black", width=1)),
                             name=f"Best (lanes={int(best_row['mac_lanes'])})"))
    fig.update_layout(
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        margin=dict(t=40, b=30, l=50, r=10), height=440,
        title=dict(text="<b>Reward Convergence</b>",
                   font=dict(color=ACCENT_CYAN, size=14), x=0.5),
        xaxis=dict(title="Trial #", color="#7090b0", gridcolor=GRID_COLOR),
        yaxis=dict(title="Reward", color="#7090b0", gridcolor=GRID_COLOR),
        legend=dict(font=dict(color="#99bbcc"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Pareto scatter ─────────────────────────────────────────────────────────────

def make_pareto(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["mac_lanes"], y=df["avg_cycles"], mode="markers",
        marker=dict(
            size=df["speedup"] * 1.5,
            color=df["reward"], colorscale="RdYlGn",
            colorbar=dict(
                title=dict(text="Reward", font=dict(color="#7090b0")),
                tickfont=dict(color="#7090b0"),
            ),
            line=dict(color="white", width=0.5),
        ),
        text=[
            f"Trial {t}<br>lanes={ml}  acc={aw}b  clk={clk}ns<br>"
            f"{cyc:,} cycles  {sp:.1f}× speedup<br>"
            f"area={ar:.2f}  slack={sl:+.1f}ns"
            for t, ml, aw, clk, cyc, sp, ar, sl in zip(
                df["trial"], df["mac_lanes"],
                df.get("accumulator_width", [32]*len(df)),
                df.get("clock_period_ns", [10]*len(df)),
                df["avg_cycles"], df["speedup"],
                df.get("area_proxy", [1.0]*len(df)),
                df.get("timing_slack_ns", [0.0]*len(df)),
            )
        ],
        hovertemplate="%{text}<extra></extra>", name="Trials",
    ))
    fig.add_hline(y=SW_BASELINE, line_dash="dash", line_color="#556677",
                  annotation_text="SW baseline (no accel)",
                  annotation_font_color="#99aabb",
                  annotation_position="top right")
    best = df.loc[df["reward"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[best["mac_lanes"]], y=[best["avg_cycles"]], mode="markers",
        marker=dict(symbol="star", size=22, color=ACCENT_GOLD,
                    line=dict(color="black", width=1)),
        name="Best",
    ))
    fig.update_layout(
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        margin=dict(t=40, b=30, l=50, r=10), height=340,
        title=dict(text="<b>Pareto Frontier</b>  ·  latency vs area",
                   font=dict(color=ACCENT_CYAN, size=14), x=0.5),
        xaxis=dict(title="MAC lanes (area proxy →)", color="#7090b0",
                   gridcolor=GRID_COLOR, tickvals=[1,2,4,8,16]),
        yaxis=dict(title="Avg cycles / inference  (↓ better)", color="#7090b0",
                   gridcolor=GRID_COLOR),
        legend=dict(font=dict(color="#99bbcc"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Speedup bars ───────────────────────────────────────────────────────────────

def make_speedup_bars(df: pd.DataFrame) -> go.Figure:
    bar_df = (df.sort_values("trial")
                .groupby("mac_lanes").last()
                .reset_index()
                .sort_values("mac_lanes"))
    colors = [ACCENT_GOLD if r == bar_df["reward"].max() else ACCENT_CYAN
              for r in bar_df["reward"]]
    fig = go.Figure(go.Bar(
        x=bar_df["mac_lanes"].astype(str), y=bar_df["speedup"],
        marker_color=colors,
        text=[f"{s:.1f}×" for s in bar_df["speedup"]],
        textposition="outside", textfont=dict(color="white", size=12),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#556677",
                  annotation_text="SW baseline",
                  annotation_font_color="#99aabb")
    fig.update_layout(
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        margin=dict(t=40, b=30, l=50, r=10), height=340,
        title=dict(text="<b>Speedup vs SW Baseline</b>",
                   font=dict(color=ACCENT_CYAN, size=14), x=0.5),
        xaxis=dict(title="MAC lanes", color="#7090b0", gridcolor=GRID_COLOR),
        yaxis=dict(title="Speedup ×", color="#7090b0", gridcolor=GRID_COLOR),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Parallel coordinates ────────────────────────────────────────────────────────

def make_parallel_coords(df: pd.DataFrame) -> go.Figure:
    """Show all dimensions at once — each line is one trial, coloured by reward."""
    def _dim(col: str, label: str, tick_vals=None, tick_text=None) -> dict:
        if col not in df.columns:
            return None
        d = dict(label=label, values=df[col].tolist(),
                 tickfont=dict(color="#99bbcc"))
        if tick_vals:
            d["tickvals"] = tick_vals
            d["ticktext"] = tick_text or [str(v) for v in tick_vals]
        return d

    # Map categorical 'dataflow' to numeric if present
    if "dataflow" in df.columns:
        df = df.copy()
        df["dataflow_num"] = df["dataflow"].map(
            {"output_stationary": 0, "weight_stationary": 1}
        ).fillna(0)

    dims = [
        _dim("mac_lanes",          "MAC Lanes", [1,2,4,8,16]),
        _dim("accumulator_width",  "Acc Width (b)", [16,24,32]),
        _dim("clock_period_ns",    "Clock (ns)", [5,10,20]),
        _dim("input_buffer_bytes", "In Buf (B)", [256,512,1024,2048,4096]),
        _dim("weight_buffer_bytes","Wt Buf (B)", [256,512,1024,2048,4096]),
        _dim("dataflow_num",       "Dataflow", [0,1], ["out-stat","wt-stat"]),
        _dim("speedup",            "Speedup ×"),
        _dim("area_proxy",         "Area proxy"),
        _dim("power_proxy",        "Power proxy"),
        _dim("timing_slack_ns",    "Timing slack (ns)"),
        _dim("reward",             "Reward"),
    ]
    dims = [d for d in dims if d is not None]

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=df["reward"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title=dict(text="Reward", font=dict(color="#7090b0")),
                          tickfont=dict(color="#7090b0")),
        ),
        dimensions=dims,
        labelfont=dict(color="#aaccdd", size=11),
    ))
    fig.update_layout(
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        margin=dict(t=50, b=30, l=60, r=60), height=340,
        title=dict(text="<b>Parallel Coordinates</b>  ·  all design dimensions",
                   font=dict(color=ACCENT_CYAN, size=14), x=0.5),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Main render loop ───────────────────────────────────────────────────────────

st.markdown(
    "<h2 style='color:#00d4ff;margin-bottom:0'>⚡ TinyMAC Design-Space Explorer</h2>"
    f"<p style='color:#7090b0;margin-top:4px'>"
    f"SW baseline (Stage 3, no accel): "
    f"<b style='color:#aaccdd'>{SW_BASELINE:,} cycles</b> per inference &nbsp;·&nbsp; "
    "optimizer finds best latency × area × timing trade-off</p>",
    unsafe_allow_html=True,
)

df = load_results()

# ── Waiting state ──────────────────────────────────────────────────────────────
if df.empty:
    st.markdown(
        "<div style='background:#161a28;border:1px solid #2a3050;border-radius:8px;"
        "padding:24px 32px;margin-top:16px'>"
        "<p style='color:#7090b0;font-size:1.1rem'>No trials yet. "
        "Start the optimizer in another terminal (WSL):</p>"
        "<pre style='color:#00d4ff;background:#0d0f18;padding:12px;border-radius:6px'>"
        "cd ~/voiceAI\n"
        "python3 optimizer/run_optimizer.py           # evo agent, 30 trials\n"
        "python3 optimizer/run_optimizer.py --agent ucb --trials 20\n"
        "python3 optimizer/run_optimizer.py --agent bayesian --trials 30"
        "</pre></div>",
        unsafe_allow_html=True,
    )
    time.sleep(REFRESH_SEC)
    st.rerun()

# ── KPI metrics ────────────────────────────────────────────────────────────────
best = df.loc[df["reward"].idxmax()]
violations = int(df.get("timing_violation", pd.Series([False]*len(df))).sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trials",          len(df))
c2.metric("Best lanes",      int(best["mac_lanes"]))
c3.metric("Best speedup",    f"{best['speedup']:.1f}×")
c4.metric("Best cycles",     f"{int(best['avg_cycles']):,}")
c5.metric("Best reward",     f"{best['reward']:.3f}")
c6.metric("Timing viols",    violations)

# ── Row 1: schematic + convergence ────────────────────────────────────────────
top_l, top_r = st.columns(2)
latest = df.iloc[-1]
with top_l:
    st.plotly_chart(make_accel_diagram(latest), use_container_width=True)
with top_r:
    st.plotly_chart(make_convergence(df), use_container_width=True)

# ── Row 2: Pareto + speedup bars ──────────────────────────────────────────────
bot_l, bot_r = st.columns(2)
with bot_l:
    st.plotly_chart(make_pareto(df), use_container_width=True)
with bot_r:
    st.plotly_chart(make_speedup_bars(df), use_container_width=True)

# ── Row 3: parallel coordinates (multi-dimensional view) ──────────────────────
if len(df) >= 3:
    st.plotly_chart(make_parallel_coords(df), use_container_width=True)
else:
    st.info("Parallel coordinates chart appears after 3+ trials.")

# ── Row 4: trial table ─────────────────────────────────────────────────────────
show_cols = [c for c in [
    "trial", "mac_lanes", "accumulator_width", "clock_period_ns",
    "avg_cycles", "speedup", "efficiency",
    "area_proxy", "power_proxy", "timing_slack_ns",
    "accuracy", "reward", "elapsed_s",
] if c in df.columns]

with st.expander("All trials", expanded=False):
    disp = (df[show_cols]
            .sort_values("reward", ascending=False)
            .reset_index(drop=True))
    if "avg_cycles" in disp:
        disp["avg_cycles"] = disp["avg_cycles"].apply(lambda x: f"{int(x):,}")
    for col in ("speedup", "efficiency"):
        if col in disp:
            disp[col] = disp[col].apply(lambda x: f"{x:.1f}×")
    for col in ("area_proxy", "power_proxy", "timing_slack_ns", "reward"):
        if col in disp:
            disp[col] = disp[col].apply(lambda x: f"{x:.3f}")
    if "accuracy" in disp:
        disp["accuracy"] = disp["accuracy"].apply(lambda x: f"{float(x):.0%}")
    if "elapsed_s" in disp:
        disp["elapsed_s"] = disp["elapsed_s"].apply(lambda x: f"{float(x):.1f}s")
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"<p style='color:#334455;font-size:0.8rem;text-align:right'>"
    f"auto-refresh every {REFRESH_SEC}s &nbsp;·&nbsp; {time.strftime('%H:%M:%S')}"
    f"&nbsp;·&nbsp; {RESULTS_FILE}</p>",
    unsafe_allow_html=True,
)
time.sleep(REFRESH_SEC)
st.rerun()

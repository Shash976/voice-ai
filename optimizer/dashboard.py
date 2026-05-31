"""dashboard.py — live Streamlit GUI for the TinyMAC design-space explorer.

Layout mirrors chip-placement visualizers:
  Top-left  : animated MAC-array schematic (chip view)
  Top-right : reward convergence curve
  Bot-left  : Pareto scatter  (cycles vs area proxy)
  Bot-right : speedup bar chart

Launch:
    streamlit run optimizer/dashboard.py

Access via SSH tunnel:
    ssh -L 8501:localhost:8501 user@vm  →  http://localhost:8501
"""

import json
import math
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_FILE  = Path(__file__).parent / "results.jsonl"
SW_BASELINE   = 175_324   # pure-SW avg cycles (Stage 3)
REFRESH_SEC   = 2
DARK_BG       = "rgba(15,17,26,1)"
PANEL_BG      = "rgba(22,26,40,1)"
GRID_COLOR    = "rgba(255,255,255,0.06)"
ACCENT_CYAN   = "#00d4ff"
ACCENT_ORANGE = "#ff6b35"
ACCENT_GREEN  = "#00ff88"
ACCENT_GOLD   = "#ffd700"

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TinyMAC Design Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark global style
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

# ── Data loading ──────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(RESULTS_FILE) as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    rows.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["efficiency"] = df["speedup"] / df["mac_lanes"]
    return df


# ── Accelerator schematic ─────────────────────────────────────────────────────

def make_accel_diagram(mac_lanes: int, avg_cycles: int, trial: int) -> go.Figure:
    """
    Draw a chip-style schematic of the TinyMAC accelerator.
    An animation phase (driven by wall-clock time) sweeps a 'computing' highlight
    through the MAC lanes so it appears live even on simple auto-refresh.
    """
    W, H = 12.0, 9.0   # canvas dimensions in arbitrary units

    # MAC grid geometry
    cols = min(mac_lanes, 8)
    rows = math.ceil(mac_lanes / cols)
    mw, mh   = 0.85, 0.65          # cell width/height
    gx, gy   = 1.15, 0.90          # cell pitch
    arr_w    = cols * gx - (gx - mw)
    arr_h    = rows * gy - (gy - mh)
    arr_x0   = (W - arr_w) / 2
    arr_y0   = 3.6

    # Animation: compute phase advances ~2× per second
    phase = int(time.time() * 2.5) % mac_lanes

    shapes, annotations, scatter_x, scatter_y, scatter_c = [], [], [], [], []

    # ── Chip die border ──
    shapes.append(dict(
        type="rect", x0=0.3, y0=0.3, x1=W - 0.3, y1=H - 0.3,
        fillcolor=PANEL_BG,
        line=dict(color=ACCENT_CYAN, width=2),
        layer="below",
    ))

    # ── Input buffer (bottom-left) ──
    shapes.append(dict(
        type="rect", x0=0.7, y0=0.7, x1=2.8, y1=2.2,
        fillcolor="rgba(0,80,140,0.6)", line=dict(color="#3399cc", width=1.5),
    ))
    annotations.append(dict(
        x=1.75, y=1.45, text="<b>Input</b><br>Buffer<br>[49×40]",
        font=dict(color="#88ccff", size=11), showarrow=False, align="center",
    ))

    # ── Weight buffer (bottom-right) ──
    shapes.append(dict(
        type="rect", x0=W - 2.8, y0=0.7, x1=W - 0.7, y1=2.2,
        fillcolor="rgba(0,80,140,0.6)", line=dict(color="#3399cc", width=1.5),
    ))
    annotations.append(dict(
        x=W - 1.75, y=1.45, text="<b>Weight</b><br>Buffer<br>[int8]",
        font=dict(color="#88ccff", size=11), showarrow=False, align="center",
    ))

    # ── Bias/quant params (bottom-center) ──
    shapes.append(dict(
        type="rect", x0=W/2 - 1.1, y0=0.7, x1=W/2 + 1.1, y1=2.2,
        fillcolor="rgba(60,40,100,0.6)", line=dict(color="#9966cc", width=1.5),
    ))
    annotations.append(dict(
        x=W / 2, y=1.45, text="<b>Bias</b><br>Q-mult<br>R-shift",
        font=dict(color="#cc99ff", size=11), showarrow=False, align="center",
    ))

    # ── Arrows: buffers → MAC array ──
    def arrow(x0, y0, x1, y1, color="#336688"):
        shapes.append(dict(
            type="line", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=color, width=1.5, dash="dot"),
        ))

    arrow(1.75, 2.2, arr_x0 + arr_w * 0.25, arr_y0)
    arrow(W - 1.75, 2.2, arr_x0 + arr_w * 0.75, arr_y0)
    arrow(W / 2, 2.2, arr_x0 + arr_w * 0.5, arr_y0)

    # ── MAC lane cells ──
    for i in range(mac_lanes):
        col = i % cols
        row = i // cols
        x0 = arr_x0 + col * gx
        y0 = arr_y0 + row * gy
        x1, y1 = x0 + mw, y0 + mh

        # Colour based on animation phase
        if i == phase:
            fill = ACCENT_ORANGE   # currently computing
            border = "#ffffff"
            lw = 2.0
        elif i < phase:
            fill = "rgba(0,180,90,0.55)"   # done
            border = ACCENT_GREEN
            lw = 1.0
        else:
            fill = "rgba(30,50,90,0.7)"   # waiting
            border = "#445577"
            lw = 1.0

        shapes.append(dict(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=fill, line=dict(color=border, width=lw),
        ))
        # Use scatter points for MAC labels (shapes can't have per-element text)
        scatter_x.append((x0 + x1) / 2)
        scatter_y.append((y0 + y1) / 2)
        scatter_c.append(i)

    # ── Output buffer (top-center) ──
    out_x0, out_x1 = W / 2 - 1.4, W / 2 + 1.4
    out_y0, out_y1 = arr_y0 + arr_h + 0.55, arr_y0 + arr_h + 1.55
    shapes.append(dict(
        type="rect", x0=out_x0, y0=out_y0, x1=out_x1, y1=out_y1,
        fillcolor="rgba(0,120,60,0.55)", line=dict(color=ACCENT_GREEN, width=1.5),
    ))
    annotations.append(dict(
        x=W / 2, y=(out_y0 + out_y1) / 2,
        text="<b>Output Buffer</b>  [int8]",
        font=dict(color=ACCENT_GREEN, size=11), showarrow=False,
    ))
    # Arrow MAC array → output
    arrow(arr_x0 + arr_w / 2, arr_y0 + arr_h, W / 2, out_y0, color=ACCENT_GREEN)

    # ── Requant unit (top-right of array) ──
    rq_x0, rq_x1 = W - 2.7, W - 0.7
    rq_y0, rq_y1 = arr_y0, arr_y0 + 0.9
    shapes.append(dict(
        type="rect", x0=rq_x0, y0=rq_y0, x1=rq_x1, y1=rq_y1,
        fillcolor="rgba(120,60,0,0.5)", line=dict(color="#ffaa44", width=1.5),
    ))
    annotations.append(dict(
        x=(rq_x0 + rq_x1) / 2, y=(rq_y0 + rq_y1) / 2,
        text="Requant<br>(per-ch)",
        font=dict(color="#ffcc88", size=10), showarrow=False,
    ))
    arrow(arr_x0 + arr_w, arr_y0 + arr_h / 2, rq_x0, (rq_y0 + rq_y1) / 2, "#ffaa44")
    arrow(rq_x0 + (rq_x1 - rq_x0) / 2, rq_y1, W / 2 + 1.4, (out_y0 + out_y1) / 2, "#ffaa44")

    # ── Stats overlay (top-left corner of die) ──
    stats = [
        f"mac_lanes : {mac_lanes}",
        f"avg cycles: {avg_cycles:,}",
        f"speedup   : {SW_BASELINE / avg_cycles:.1f}×",
        f"trial     : {trial}",
        f"phase     : {phase}/{mac_lanes - 1}",
    ]
    for j, line in enumerate(stats):
        annotations.append(dict(
            x=0.55, y=H - 0.55 - j * 0.42,
            text=f"<span style='font-family:monospace;font-size:11px'>{line}</span>",
            font=dict(color="#99bbcc", size=11),
            showarrow=False, xanchor="left",
        ))

    # ── MAC label scatter ──
    fig = go.Figure()
    if scatter_x:
        fig.add_trace(go.Scatter(
            x=scatter_x, y=scatter_y,
            mode="text",
            text=[f"M{i}" for i in range(mac_lanes)],
            textfont=dict(color="white", size=9),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(range=[0, W], visible=False, fixedrange=True),
        yaxis=dict(range=[0, H], visible=False, fixedrange=True, scaleanchor="x"),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        margin=dict(t=40, b=8, l=8, r=8),
        height=440,
        title=dict(
            text=(f"<b>TinyMAC Accelerator</b>  ·  "
                  f"{mac_lanes} MAC lane{'s' if mac_lanes > 1 else ''}  ·  "
                  f"Trial {trial}"),
            font=dict(color=ACCENT_CYAN, size=14),
            x=0.5,
        ),
    )
    return fig


# ── Convergence curve ─────────────────────────────────────────────────────────

def make_convergence(df: pd.DataFrame) -> go.Figure:
    df_s = df.sort_values("trial")
    df_s["best_so_far"] = df_s["reward"].cummax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_s["trial"], y=df_s["reward"],
        mode="markers+lines",
        name="Trial reward",
        line=dict(color="#4477ff", width=1.5, dash="dot"),
        marker=dict(size=8, color="#4477ff",
                    line=dict(color="white", width=1)),
    ))
    fig.add_trace(go.Scatter(
        x=df_s["trial"], y=df_s["best_so_far"],
        mode="lines",
        name=f"Best {df_s['best_so_far'].iloc[-1]:.3f}",
        line=dict(color=ACCENT_GREEN, width=2.5),
    ))
    # Best marker
    best_row = df_s.loc[df_s["reward"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[best_row["trial"]], y=[best_row["reward"]],
        mode="markers",
        marker=dict(symbol="star", size=18, color=ACCENT_GOLD,
                    line=dict(color="black", width=1)),
        name=f"Best (lanes={int(best_row['mac_lanes'])})",
    ))
    fig.update_layout(
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        margin=dict(t=40, b=30, l=50, r=10),
        height=440,
        title=dict(text="<b>Reward Convergence</b>",
                   font=dict(color=ACCENT_CYAN, size=14), x=0.5),
        xaxis=dict(title="Trial #", color="#7090b0",
                   gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(title="Reward (speedup / area)", color="#7090b0",
                   gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(font=dict(color="#99bbcc"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Pareto scatter ────────────────────────────────────────────────────────────

def make_pareto(df: pd.DataFrame, best_ml: int) -> go.Figure:
    fig = go.Figure()

    # All trials
    fig.add_trace(go.Scatter(
        x=df["mac_lanes"], y=df["avg_cycles"],
        mode="markers",
        marker=dict(
            size=df["speedup"] * 2,
            color=df["reward"],
            colorscale="RdYlGn",
            colorbar=dict(title="Reward", tickfont=dict(color="#7090b0"),
                          titlefont=dict(color="#7090b0")),
            line=dict(color="white", width=0.5),
        ),
        text=[f"Trial {t}<br>lanes={ml}<br>{cyc:,} cycles<br>{sp:.1f}× speedup"
              for t, ml, cyc, sp in zip(df["trial"], df["mac_lanes"],
                                        df["avg_cycles"], df["speedup"])],
        hovertemplate="%{text}<extra></extra>",
        name="Trials",
    ))

    # SW baseline line
    fig.add_hline(y=SW_BASELINE, line_dash="dash", line_color="#556677",
                  annotation_text="SW baseline (no accel)",
                  annotation_font_color="#99aabb",
                  annotation_position="top right")

    # Best star
    best = df.loc[df["reward"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[best["mac_lanes"]], y=[best["avg_cycles"]],
        mode="markers",
        marker=dict(symbol="star", size=22, color=ACCENT_GOLD,
                    line=dict(color="black", width=1)),
        name="Best",
    ))

    fig.update_layout(
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        margin=dict(t=40, b=30, l=50, r=10),
        height=340,
        title=dict(text="<b>Pareto Frontier</b>  ·  latency vs area",
                   font=dict(color=ACCENT_CYAN, size=14), x=0.5),
        xaxis=dict(title="MAC lanes (area proxy →)", color="#7090b0",
                   gridcolor=GRID_COLOR, tickvals=[1, 2, 4, 8, 16]),
        yaxis=dict(title="Avg cycles / inference  (↓ better)", color="#7090b0",
                   gridcolor=GRID_COLOR),
        legend=dict(font=dict(color="#99bbcc"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Speedup bars ──────────────────────────────────────────────────────────────

def make_speedup_bars(df: pd.DataFrame) -> go.Figure:
    bar_df = (df.sort_values("trial")
                .groupby("mac_lanes").last()
                .reset_index()
                .sort_values("mac_lanes"))

    colors = [ACCENT_GOLD if r == bar_df["reward"].max() else ACCENT_CYAN
              for r in bar_df["reward"]]

    fig = go.Figure(go.Bar(
        x=bar_df["mac_lanes"].astype(str),
        y=bar_df["speedup"],
        marker_color=colors,
        text=[f"{s:.1f}×" for s in bar_df["speedup"]],
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#556677",
                  annotation_text="SW baseline",
                  annotation_font_color="#99aabb")
    fig.update_layout(
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        margin=dict(t=40, b=30, l=50, r=10),
        height=340,
        title=dict(text="<b>Speedup vs SW Baseline</b>",
                   font=dict(color=ACCENT_CYAN, size=14), x=0.5),
        xaxis=dict(title="MAC lanes", color="#7090b0", gridcolor=GRID_COLOR),
        yaxis=dict(title="Speedup ×", color="#7090b0", gridcolor=GRID_COLOR),
        font=dict(color="#99bbcc"),
    )
    return fig


# ── Main render loop ──────────────────────────────────────────────────────────

st.markdown(
    "<h2 style='color:#00d4ff;margin-bottom:0'>⚡ TinyMAC Design-Space Explorer</h2>"
    f"<p style='color:#7090b0;margin-top:4px'>SW baseline (Stage 3, no accel): "
    f"<b style='color:#aaccdd'>{SW_BASELINE:,} cycles</b> per inference &nbsp;·&nbsp; "
    "optimizer finds best latency × area trade-off</p>",
    unsafe_allow_html=True,
)

df = load_results()

# ── Waiting state ─────────────────────────────────────────────────────────────
if df.empty:
    st.markdown(
        "<div style='background:#161a28;border:1px solid #2a3050;border-radius:8px;"
        "padding:24px 32px;margin-top:16px'>"
        "<p style='color:#7090b0;font-size:1.1rem'>No trials yet. "
        "Start the optimizer in another terminal:</p>"
        "<pre style='color:#00d4ff;background:#0d0f18;padding:12px;border-radius:6px'>"
        "cd ~/voiceAI\npython3 optimizer/search.py</pre></div>",
        unsafe_allow_html=True,
    )
    time.sleep(REFRESH_SEC)
    st.rerun()

# ── Metrics row ───────────────────────────────────────────────────────────────
best = df.loc[df["reward"].idxmax()]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trials",         len(df))
c2.metric("Best mac_lanes", int(best["mac_lanes"]))
c3.metric("Best speedup",   f"{best['speedup']:.1f}×")
c4.metric("Best cycles",    f"{int(best['avg_cycles']):,}")
c5.metric("Best reward",    f"{best['reward']:.3f}")

# ── Top row: accel diagram + convergence ──────────────────────────────────────
top_l, top_r = st.columns(2)

latest = df.iloc[-1]   # most recently completed trial
with top_l:
    st.plotly_chart(
        make_accel_diagram(
            mac_lanes  = int(latest["mac_lanes"]),
            avg_cycles = int(latest["avg_cycles"]),
            trial      = int(latest["trial"]),
        ),
        use_container_width=True,
    )

with top_r:
    st.plotly_chart(make_convergence(df), use_container_width=True)

# ── Bottom row: Pareto + speedup bars ────────────────────────────────────────
bot_l, bot_r = st.columns(2)
with bot_l:
    st.plotly_chart(make_pareto(df, int(best["mac_lanes"])), use_container_width=True)
with bot_r:
    st.plotly_chart(make_speedup_bars(df), use_container_width=True)

# ── Trial table ───────────────────────────────────────────────────────────────
with st.expander("All trials", expanded=False):
    disp = (
        df[["trial", "mac_lanes", "avg_cycles", "speedup", "efficiency", "reward",
            "accuracy", "elapsed_s"]]
        .sort_values("reward", ascending=False)
        .reset_index(drop=True)
    )
    disp["avg_cycles"] = disp["avg_cycles"].apply(lambda x: f"{x:,}")
    disp["speedup"]    = disp["speedup"].apply(lambda x: f"{x:.1f}×")
    disp["efficiency"] = disp["efficiency"].apply(lambda x: f"{x:.2f}")
    disp["reward"]     = disp["reward"].apply(lambda x: f"{x:.3f}")
    disp["accuracy"]   = disp["accuracy"].apply(lambda x: f"{x:.0%}")
    disp["elapsed_s"]  = disp["elapsed_s"].apply(lambda x: f"{x:.1f}s")
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown(
    f"<p style='color:#334455;font-size:0.8rem;text-align:right'>"
    f"auto-refresh every {REFRESH_SEC}s &nbsp;·&nbsp; {time.strftime('%H:%M:%S')}</p>",
    unsafe_allow_html=True,
)
time.sleep(REFRESH_SEC)
st.rerun()

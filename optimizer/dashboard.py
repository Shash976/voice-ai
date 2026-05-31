"""dashboard.py — live Streamlit GUI for the TinyMAC design-space explorer.

Launch:
    streamlit run optimizer/dashboard.py

Access:
    http://localhost:8501  (or via SSH tunnel: ssh -L 8501:localhost:8501 user@vm)

The page auto-refreshes every 3 seconds so you can watch trials come in live
while search.py runs in a separate terminal.
"""

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_FILE = Path(__file__).parent / "results.jsonl"
SW_BASELINE  = 175_324   # pure-SW avg cycles (Stage 3, no accelerator)
REFRESH_SEC  = 3


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["area_proxy"] = df["mac_lanes"]          # area ∝ lane count
    df["efficiency"] = df["speedup"] / df["mac_lanes"]
    return df


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TinyMAC Design Explorer",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ TinyMAC Design-Space Explorer")
st.caption(
    f"SW baseline (no accel): **{SW_BASELINE:,} cycles/inference** — "
    "optimizer searches for the best latency × area trade-off."
)

df = load_results()

# ── Empty state ───────────────────────────────────────────────────────────────

if df.empty:
    st.info(
        "No results yet. Start the optimizer in another terminal:\n\n"
        "```bash\n"
        "cd ~/voiceAI\n"
        "python optimizer/search.py\n"
        "```"
    )
    time.sleep(REFRESH_SEC)
    st.rerun()

# ── Top metrics row ───────────────────────────────────────────────────────────

best = df.loc[df["reward"].idxmax()]
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Trials completed",  len(df))
col2.metric("Best mac_lanes",    int(best["mac_lanes"]))
col3.metric("Best speedup",      f"{best['speedup']:.1f}×")
col4.metric("Best avg cycles",   f"{int(best['avg_cycles']):,}")
col5.metric("Best reward",       f"{best['reward']:.3f}")

st.divider()

# ── Main charts ───────────────────────────────────────────────────────────────

left, right = st.columns(2)

# Pareto scatter: cycles vs mac_lanes, colour = reward
with left:
    st.subheader("Pareto frontier — latency vs. area")
    fig = px.scatter(
        df,
        x="mac_lanes",
        y="avg_cycles",
        color="reward",
        size="speedup",
        hover_data=["trial", "speedup", "accuracy", "elapsed_s"],
        color_continuous_scale="RdYlGn",
        labels={
            "mac_lanes":  "MAC lanes (area proxy →)",
            "avg_cycles": "Avg cycles / inference (latency ↓)",
            "reward":     "Reward",
        },
    )
    # Add SW baseline reference line
    fig.add_hline(
        y=SW_BASELINE,
        line_dash="dash",
        line_color="grey",
        annotation_text="SW baseline",
        annotation_position="top right",
    )
    # Highlight best point
    fig.add_trace(go.Scatter(
        x=[best["mac_lanes"]],
        y=[best["avg_cycles"]],
        mode="markers",
        marker=dict(symbol="star", size=18, color="gold",
                    line=dict(color="black", width=1)),
        name="Best",
        showlegend=True,
    ))
    fig.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# Speedup bar chart
with right:
    st.subheader("Speedup over SW baseline")
    # One bar per mac_lanes (take last result if repeated)
    bar_df = df.sort_values("trial").groupby("mac_lanes").last().reset_index()
    fig2 = px.bar(
        bar_df,
        x="mac_lanes",
        y="speedup",
        color="speedup",
        text=bar_df["speedup"].apply(lambda s: f"{s:.1f}×"),
        color_continuous_scale="Blues",
        labels={"mac_lanes": "MAC lanes", "speedup": "Speedup vs SW"},
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        height=400,
        margin=dict(t=30, b=10),
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Reward over time ──────────────────────────────────────────────────────────

st.subheader("Reward over trials")
df_sorted = df.sort_values("trial")
df_sorted["best_so_far"] = df_sorted["reward"].cummax()
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=df_sorted["trial"], y=df_sorted["reward"],
    mode="markers+lines", name="Trial reward",
    line=dict(color="#636EFA", dash="dot"), marker=dict(size=8),
))
fig3.add_trace(go.Scatter(
    x=df_sorted["trial"], y=df_sorted["best_so_far"],
    mode="lines", name="Best so far",
    line=dict(color="#00CC96", width=3),
))
fig3.update_layout(
    height=280,
    margin=dict(t=10, b=10),
    xaxis_title="Trial #",
    yaxis_title="Reward (speedup / area)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig3, use_container_width=True)

# ── Efficiency table ──────────────────────────────────────────────────────────

st.subheader("All trials")
display_df = (
    df[["trial", "mac_lanes", "avg_cycles", "speedup", "efficiency", "reward", "accuracy", "elapsed_s"]]
    .sort_values("reward", ascending=False)
    .reset_index(drop=True)
)
display_df["avg_cycles"] = display_df["avg_cycles"].apply(lambda x: f"{x:,}")
display_df["speedup"]    = display_df["speedup"].apply(lambda x: f"{x:.1f}×")
display_df["efficiency"] = display_df["efficiency"].apply(lambda x: f"{x:.2f}")
display_df["reward"]     = display_df["reward"].apply(lambda x: f"{x:.3f}")
display_df["accuracy"]   = display_df["accuracy"].apply(lambda x: f"{x:.0%}")
display_df["elapsed_s"]  = display_df["elapsed_s"].apply(lambda x: f"{x:.1f}s")
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

st.caption(f"Auto-refreshing every {REFRESH_SEC}s · {time.strftime('%H:%M:%S')}")
time.sleep(REFRESH_SEC)
st.rerun()

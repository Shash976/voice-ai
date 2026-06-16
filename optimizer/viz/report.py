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
)

_FIDELITY_COLORS = {
    "F0": "#d62728",   # red — died at proxy gate 0
    "F1": "#ff7f0e",
    "F2": "#9467bd",
    "F3": "#2ca02c",   # green — reached full flow
    None: "#7f7f7f",
}


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


def write_html(figs, out_path: Path, title: str):
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:0;background:#fafafa;}"
        "h1{padding:16px 24px;margin:0;background:#1f2937;color:#fff;font-size:20px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));"
        "gap:8px;padding:8px;}.card{background:#fff;border:1px solid #e5e7eb;"
        "border-radius:8px;padding:4px;}</style></head><body>",
        f"<h1>{title}</h1><div class='grid'>",
    ]
    first = True
    for _label, fig in figs:
        frag = fig.to_html(full_html=False,
                           include_plotlyjs=("cdn" if first else False),
                           default_height="420px")
        parts.append(f"<div class='card'>{frag}</div>")
        first = False
    parts.append("</div></body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Static HTML report for a campaign log")
    ap.add_argument("--log", default=str(DEFAULT_LOG),
                    help="campaign JSONL (default: results_funnel_campaigns.jsonl)")
    ap.add_argument("--campaign", default="latest",
                    help="campaign_id | 'latest' | 'all'")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--no-optuna", action="store_true",
                    help="skip Optuna importance/slice/contour figures")
    ap.add_argument("--open", action="store_true", help="open the report in a browser")
    args = ap.parse_args()

    data = CampaignData.load(args.log, args.campaign)
    if not data.rows:
        print(f"No episodes found in {args.log} for campaign={args.campaign!r}")
        sys.exit(1)

    print(f"Loaded {len(data.rows)} episodes (campaign={data.campaign_id}, "
          f"{len(data.specs)} params)")

    figs = build_figures(data)
    if not args.no_optuna:
        figs += build_optuna_figures(data.rows)

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / f"report_{data.campaign_id}.html"
    title = f"Funnel optimizer · {data.campaign_id} · {len(data.rows)} episodes"
    write_html(figs, out, title)
    print(f"Wrote {len(figs)} figures → {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()

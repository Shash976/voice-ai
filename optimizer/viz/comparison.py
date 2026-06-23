#!/usr/bin/env python3
"""comparison.py — self-contained before/after chip layout comparison page.

    python3 optimizer/viz/comparison.py
    python3 optimizer/viz/comparison.py --out /tmp/compare.html --open

Generates a single HTML file embedding four layout images (base64) and their
measured physical metrics, showing the baseline hand-picked design versus three
Pareto-optimal designs found by the funnel optimizer.
"""

from __future__ import annotations

import argparse
import base64
import sys
import webbrowser
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_REPORTS = _REPO / "optimizer" / "reports"
_ORFS_RESULTS = _REPO / "physical" / "orfs" / "make" / "results" / "asap7" / "tinymac_accel"
_LYP = Path("/opt/OpenROAD-flow-scripts/flow/platforms/asap7/KLayout/asap7.lyp")
_RENDER_SCRIPT = Path(__file__).parent / "_render_gds.rb"

_SW_LATENCY_NS = 11_196_638 * 10.0   # Stage-3 SW @ 100 MHz
_AVG_CYCLES = {1: 273_130, 2: 152_140, 4: 91_650, 8: 61_400, 16: 46_670, 32: 39_310}

DESIGNS = [
    {
        "key":      "baseline",
        "role":     "before",
        "badge":    "HAND-PICKED",
        "label":    "Baseline",
        "sublabel": "Stage 6 — first asap7 GDS",
        "variant":  "L4_A24_c1_rb8d8aaa8",
        "mac_lanes": 4, "acc_w": 24, "clk_ns": 1.0, "recipe": "orfs_speed",
        "area_um2": 1433, "fmax_mhz": 509, "wns_ns": -0.96,
        "image":    "layout_baseline.png",
    },
    {
        "key":      "best_speedup",
        "role":     "after",
        "badge":    "BEST SPEEDUP",
        "label":    "L8_A24 · 8 lanes",
        "sublabel": "Optimizer pick — max inference speed",
        "variant":  "L8_A24_c1p104_area_rb8d8aaa8",
        "mac_lanes": 8, "acc_w": 24, "clk_ns": 1.104, "recipe": "orfs_area",
        "area_um2": 1828, "fmax_mhz": 565, "wns_ns": -0.665,
        "image":    "layout_best_speedup.png",
        "highlight": True,
    },
    {
        "key":      "best_fmax",
        "role":     "after",
        "badge":    "BEST Fmax",
        "label":    "L2_A24 · 2 lanes",
        "sublabel": "Optimizer pick — max clock frequency",
        "variant":  "L2_A24_c0p6987_area_rb8d8aaa8",
        "mac_lanes": 2, "acc_w": 24, "clk_ns": 0.699, "recipe": "orfs_area",
        "area_um2": 1484, "fmax_mhz": 594, "wns_ns": -0.984,
        "image":    "layout_best_fmax.png",
    },
    {
        "key":      "min_area",
        "role":     "after",
        "badge":    "MIN AREA",
        "label":    "L1_A24 · 1 lane",
        "sublabel": "Optimizer pick — smallest die",
        "variant":  "L1_A24_c1p225_rb8d8aaa8",
        "mac_lanes": 1, "acc_w": 24, "clk_ns": 1.225, "recipe": "orfs_speed",
        "area_um2": 1197, "fmax_mhz": 479, "wns_ns": -0.619,
        "image":    "layout_min_area.png",
    },
]


def _speedup(d: dict) -> float:
    cyc = _AVG_CYCLES.get(d["mac_lanes"], 91_650)
    period_ns = 1000.0 / d["fmax_mhz"]
    return _SW_LATENCY_NS / (cyc * period_ns)


def _latency_ms(d: dict) -> float:
    cyc = _AVG_CYCLES.get(d["mac_lanes"], 91_650)
    period_ns = 1000.0 / d["fmax_mhz"]
    return cyc * period_ns / 1e6


def _render_gds(variant: str, out_png: Path, size: int = 1800) -> bool:
    """Render a GDS layout to PNG using KLayout batch mode. Returns True on success."""
    import subprocess, shutil
    if not shutil.which("klayout"):
        return False
    gds = _ORFS_RESULTS / variant / "6_final.gds"
    if not gds.exists():
        return False

    rb = """
view = RBA::LayoutView.new
view.load_layer_props($lyp_file) if $lyp_file && File.exist?($lyp_file)
view.load_layout($gds_file)
view.max_hier
view.zoom_fit
view.save_image($out_png, Integer($width), Integer($height))
"""
    script = Path("/tmp/_klayout_render.rb")
    script.write_text(rb)
    result = subprocess.run(
        ["klayout", "-z",
         "-rd", f"gds_file={gds}",
         "-rd", f"lyp_file={_LYP}",
         "-rd", f"out_png={out_png}",
         "-rd", f"width={size}",
         "-rd", f"height={size}",
         "-r", str(script)],
        capture_output=True, text=True,
    )
    return out_png.exists()


def _png_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _delta_html(val: float, baseline: float, higher_is_better: bool = True) -> str:
    pct = (val - baseline) / baseline * 100
    better = (pct > 0) == higher_is_better
    color = "#059669" if better else "#dc2626"
    sign = "+" if pct > 0 else ""
    return f'<span style="color:{color};font-weight:700">{sign}{pct:.1f}%</span>'


def _stat_row(label: str, val: str, delta_html: str = "") -> str:
    return (
        f"<tr><td class='stat-label'>{label}</td>"
        f"<td class='stat-val'>{val}</td>"
        f"<td class='stat-delta'>{delta_html}</td></tr>"
    )


def _card_html(d: dict, baseline: dict | None, img_b64: str | None) -> str:
    is_before = d["role"] == "before"
    is_highlight = d.get("highlight", False)

    if is_before:
        border_color = "#b45309"       # amber — "before"
        badge_bg     = "#92400e"
        header_bg    = "#1c1917"
    elif is_highlight:
        border_color = "#059669"       # emerald — primary recommendation
        badge_bg     = "#065f46"
        header_bg    = "#052e16"
    else:
        border_color = "#2563eb"       # blue — secondary
        badge_bg     = "#1e3a8a"
        header_bg    = "#0f172a"

    sp     = _speedup(d)
    lat    = _latency_ms(d)
    b_sp   = _speedup(baseline) if baseline else sp
    b_area = baseline["area_um2"] if baseline else d["area_um2"]
    b_fmax = baseline["fmax_mhz"] if baseline else d["fmax_mhz"]

    img_tag = (
        f'<img src="data:image/png;base64,{img_b64}" '
        f'style="width:100%;height:auto;display:block;border-radius:6px;'
        f'border:1px solid #334155;image-rendering:pixelated" '
        f'alt="{d["label"]} layout">'
        if img_b64 else
        '<div style="width:100%;aspect-ratio:1;background:#1e293b;border-radius:6px;'
        'display:flex;align-items:center;justify-content:center;color:#64748b;font-size:13px">'
        'Layout image not available</div>'
    )

    rows = [
        _stat_row("Config",    f'L{d["mac_lanes"]}_A{d["acc_w"]}'),
        _stat_row("MAC Lanes", str(d["mac_lanes"]),
                  "" if not baseline else _delta_html(d["mac_lanes"], baseline["mac_lanes"])),
        _stat_row("Acc Width", f'{d["acc_w"]} bits'),
        _stat_row("Clock (target)", f'{d["clk_ns"]:.3f} ns'),
        _stat_row("ABC Recipe", d["recipe"]),
        "<tr><td colspan='3' style='height:6px'></td></tr>",
        _stat_row("Area",  f'{d["area_um2"]:,} µm²',
                  "" if not baseline else _delta_html(d["area_um2"], b_area, higher_is_better=False)),
        _stat_row("Fmax",  f'{d["fmax_mhz"]:.0f} MHz',
                  "" if not baseline else _delta_html(d["fmax_mhz"], b_fmax)),
        _stat_row("WNS",   f'{d["wns_ns"]:.3f} ns'),
        _stat_row("Timing", "❌ not met (all configs)"),
        "<tr><td colspan='3' style='height:6px'></td></tr>",
        _stat_row("Speedup vs SW", f'<b>{sp:.0f}×</b>',
                  "" if not baseline else _delta_html(sp, b_sp)),
        _stat_row("Latency @ Fmax", f'{lat:.3f} ms'),
    ]

    return f"""
<div class="chip-card" style="border:2px solid {border_color}">
  <div class="card-header" style="background:{header_bg}">
    <span class="badge" style="background:{badge_bg}">{d['badge']}</span>
    <div class="card-title">{d['label']}</div>
    <div class="card-sub">{d['sublabel']}</div>
  </div>
  <div class="layout-img">{img_tag}</div>
  <table class="stats-table">
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>"""


_PAGE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
}
header {
  background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
  padding: 28px 32px 20px;
  border-bottom: 1px solid #1e293b;
}
header h1 {
  font-size: 24px;
  font-weight: 800;
  letter-spacing: -0.5px;
  color: #f1f5f9;
}
header .h1-sub {
  font-size: 13px;
  color: #64748b;
  margin-top: 5px;
}
.divider-label {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 32px 8px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #475569;
}
.divider-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #1e293b;
}
.cards-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 16px;
  padding: 8px 32px 32px;
}
.chip-card {
  background: #1e293b;
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.chip-card.before {
  grid-column: 1 / 2;
}
.card-header {
  padding: 14px 16px 12px;
}
.badge {
  display: inline-block;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #fff;
  padding: 3px 8px;
  border-radius: 4px;
  margin-bottom: 8px;
}
.card-title {
  font-size: 16px;
  font-weight: 700;
  color: #f8fafc;
}
.card-sub {
  font-size: 11px;
  color: #64748b;
  margin-top: 2px;
}
.layout-img {
  padding: 10px 12px 6px;
}
.stats-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 6px;
  font-size: 12px;
}
.stats-table td {
  padding: 4px 12px;
  color: #cbd5e1;
  vertical-align: middle;
}
.stat-label {
  color: #64748b;
  width: 42%;
}
.stat-val {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11.5px;
  color: #e2e8f0;
  width: 35%;
}
.stat-delta {
  font-size: 11px;
  text-align: right;
  width: 23%;
}
.stats-table tr:hover td { background: rgba(255,255,255,0.03); }
.footer-bar {
  background: #0f172a;
  border-top: 1px solid #1e293b;
  padding: 16px 32px;
  display: flex;
  gap: 32px;
  align-items: center;
  flex-wrap: wrap;
}
.footer-stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.footer-stat .fval {
  font-size: 20px;
  font-weight: 700;
  color: #38bdf8;
}
.footer-stat .flabel {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.legend {
  margin-left: auto;
  font-size: 11px;
  color: #475569;
  text-align: right;
  line-height: 1.8;
}
"""


def build_page(designs: list[dict], report_dir: Path) -> str:
    baseline = next(d for d in designs if d["role"] == "before")
    img_b64s = {}
    for d in designs:
        png = report_dir / d["image"]
        if not png.exists():
            print(f"  Rendering {d['variant']}…", flush=True)
            _render_gds(d["variant"], png, size=1600)
        if png.exists():
            img_b64s[d["key"]] = _png_to_b64(png)
        else:
            img_b64s[d["key"]] = None
            print(f"  [warn] image not found: {png}")

    cards_html = ""
    for d in designs:
        b = None if d["role"] == "before" else baseline
        cards_html += _card_html(d, b, img_b64s[d["key"]])

    # Footer stats
    b_sp = _speedup(baseline)
    best_sp = max(_speedup(d) for d in designs)
    best_fmax = max(d["fmax_mhz"] for d in designs)
    min_area = min(d["area_um2"] for d in designs)
    n_after = sum(1 for d in designs if d["role"] == "after")

    footer = f"""
<div class="footer-bar">
  <div class="footer-stat">
    <span class="fval">{best_sp:.0f}×</span>
    <span class="flabel">Peak speedup vs software</span>
  </div>
  <div class="footer-stat">
    <span class="fval">+{(best_sp - b_sp) / b_sp * 100:.0f}%</span>
    <span class="flabel">Speedup gain vs baseline</span>
  </div>
  <div class="footer-stat">
    <span class="fval">{best_fmax:.0f} MHz</span>
    <span class="flabel">Peak Fmax found</span>
  </div>
  <div class="footer-stat">
    <span class="fval">{min_area:,} µm²</span>
    <span class="flabel">Minimum area found</span>
  </div>
  <div class="footer-stat">
    <span class="fval">33</span>
    <span class="flabel">Full P&amp;R builds</span>
  </div>
  <div class="legend">
    asap7 7nm PDK &nbsp;·&nbsp; OpenROAD-flow-scripts<br>
    18 search variables &nbsp;·&nbsp; 5.77 h wall-clock<br>
    All configs: timing not met (wns &lt; 0)
  </div>
</div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TinyMAC Accelerator — Before vs After Optimization</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<header>
  <h1>TinyMAC Accelerator &nbsp;·&nbsp; Before vs After Optimization</h1>
  <p class="h1-sub">asap7 7nm PDK &nbsp;·&nbsp; ORFS Physical Design &nbsp;·&nbsp;
  Multi-fidelity funnel optimizer &nbsp;·&nbsp; Chip layouts rendered from 6_final.gds</p>
</header>
<div class="divider-label">Layout comparison — 1 baseline vs {n_after} optimizer-found designs</div>
<div class="cards-grid">
{cards_html}
</div>
{footer}
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Before/after chip comparison page")
    ap.add_argument("--out", default=str(_REPORTS / "comparison.html"))
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Building comparison page…")
    html = build_page(DESIGNS, _REPORTS)
    out.write_text(html, encoding="utf-8")
    size_kb = len(html.encode()) / 1024
    print(f"Wrote {size_kb:.0f} KB → {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()

"""physical_runner.py — run one real ORFS RTL→GDS trial and parse its metrics.

The Stage-6 analogue of runner.py (which drives Verilator).  Where runner.py
returns cycle counts from the behavioral sim, this drives the *classic
OpenROAD-flow-scripts make flow* on a machine that has a real OpenROAD (the
company VM at /opt/OpenROAD-flow-scripts) and returns real physical metrics:
area, WNS/TNS, Fmax, power.

It reuses the exact per-config mechanism of physical/orfs/make/sweep.sh:
  - VERILOG_TOP_PARAMS="LANES n ACC_W w"   → ORFS chparam overrides RTL params
  - FLOW_VARIANT=<variant>                 → each config gets its own results dir
  - a generated SDC                        → sets the clock period

Report strings parsed below were confirmed against a real VM run:
  reports/<plat>/<design>/<variant>/6_finish.rpt :
      "wns max -1.72"      (value is the LAST field)
      "tns max -25.86"
      "core_clock period_min = 3.72 fmax = 268.64"
      "setup violation count 40"
      report_power "Total  5.46e-01 4.69e-01 4.38e-04 1.02e+00 100.0%"  (total W = field 5)
  logs/<plat>/<design>/<variant>/6_report.log :
      "Design area 19738 um^2 48% utilization."

Set PHYSICAL_MOCK=1 to skip OpenROAD and return synthetic-but-plausible metrics
(for testing the optimizer loop on a machine without the tools).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

_REPO    = Path(__file__).resolve().parent.parent
MAKE_DIR = _REPO / "physical" / "orfs" / "make"
RTL_DIR  = _REPO / "rtl" / "accel"
DESIGN   = "tinymac_accel"
RTL_FILES = ("tinymac_accel.v", "int8_mac_array.v", "requantize.v")

ORFS_DIR     = Path(os.environ.get("ORFS_DIR", "/opt/OpenROAD-flow-scripts"))
ORFS_TIMEOUT = int(os.environ.get("ORFS_TIMEOUT", "2400"))   # seconds; P&R is slow
PROXY_TIMEOUT = int(os.environ.get("PROXY_TIMEOUT", "300"))  # synth+STA is fast

# Single std-cell liberty per platform, for the fast synth+STA proxy.
# (asap7 ships many gzipped per-cell-type libs — not wired for the proxy.)
_LIBERTY = {
    "nangate45": "platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib",
    "sky130hd":  "platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
}
# Tech + std-cell LEF (OpenROAD's link_design needs a technology, not just
# liberty).  Curated — deliberately excludes the SRAM/fakeram macro LEFs.
_LEF = {
    "nangate45": ["platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef",
                  "platforms/nangate45/lef/NangateOpenCellLibrary.macro.lef"],
    "sky130hd":  ["platforms/sky130hd/lef/sky130_fd_sc_hd.tlef",
                  "platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef"],
}
# Synth cell area → estimated post-P&R "Design area" inflation (CTS/repair
# buffers). Calibrated on nangate45 LANES=4: 19738 / 14589 ≈ 1.35.
_PLACE_INFLATION = 1.35


def variant_name(lanes: int, acc_w: int, clk_ns: float) -> str:
    # clk formatted to one decimal so int/float inputs collapse to one name and
    # it matches sweep.sh's variant naming (which normalises the same way).
    return f"L{lanes}_A{acc_w}_c{f'{float(clk_ns):.1f}'.replace('.', 'p')}"


# ── Report parsing ────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _last_float_on_line(text: str, prefix: str) -> float | None:
    """Return the last numeric token of the first line starting with `prefix`.
    Matches OpenROAD lines like 'wns max -1.72' / 'tns max -25.86'."""
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", line)
            if nums:
                return float(nums[-1])
    return None


def _parse_metrics(work: Path, platform: str, variant: str, clk_ns: float) -> dict:
    rpt  = work / "reports" / platform / DESIGN / variant / "6_finish.rpt"
    rlog = work / "logs"    / platform / DESIGN / variant / "6_report.log"
    gds  = work / "results" / platform / DESIGN / variant / "6_final.gds"

    rpt_txt, rlog_txt = _read(rpt), _read(rlog)

    out: dict = {
        "area_um2": None, "util_pct": None, "wns_ns": None, "tns_ns": None,
        "setup_viol": None, "power_mw": None, "fmax_mhz": None,
        "period_min_ns": None, "timing_met": None,
        "gds": str(gds) if gds.exists() else None,
        "report": str(rpt) if rpt.exists() else None,
    }

    # area + utilisation (from the report LOG): "Design area 19738 um^2 48% utilization."
    m = re.search(r"Design area\s+([\d.]+)\s+um\^2\s+([\d.]+)%", rlog_txt)
    if m:
        out["area_um2"] = float(m.group(1))
        out["util_pct"] = float(m.group(2))

    # timing (from the finish RPT): "wns max -1.72" / "tns max -25.86"
    out["wns_ns"] = _last_float_on_line(rpt_txt, "wns")
    out["tns_ns"] = _last_float_on_line(rpt_txt, "tns")

    # Fmax + min period: "core_clock period_min = 3.72 fmax = 268.64"
    m = re.search(r"period_min\s*=\s*([\d.]+).*?fmax\s*=\s*([\d.]+)", rpt_txt)
    if m:
        out["period_min_ns"] = float(m.group(1))
        out["fmax_mhz"]      = float(m.group(2))

    m = re.search(r"setup violation count\s+(\d+)", rpt_txt)
    if m:
        out["setup_viol"] = int(m.group(1))

    # total power (report_power "Total" row, 5th column = Total Watts)
    for line in rpt_txt.splitlines():
        if line.strip().startswith("Total"):
            nums = re.findall(r"\d+\.?\d*(?:[eE][-+]?\d+)?", line)
            if len(nums) >= 4:                      # internal, switching, leakage, total[, %]
                out["power_mw"] = float(nums[3]) * 1000.0
            break

    # timing met: prefer the explicit violation count, else sign of WNS
    if out["setup_viol"] is not None:
        out["timing_met"] = (out["setup_viol"] == 0)
    elif out["wns_ns"] is not None:
        out["timing_met"] = (out["wns_ns"] >= 0.0)

    return out


# ── ORFS config generation (mirrors sweep.sh) ─────────────────────────────────

def _config_mk(platform: str, variant: str, lanes: int, acc_w: int) -> str:
    return (
        "export DESIGN_HOME = .\n"
        f"export DESIGN_NAME = {DESIGN}\n"
        f"export PLATFORM    = {platform}\n"
        "export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/tinymac_accel.v \\\n"
        "                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/int8_mac_array.v \\\n"
        "                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/requantize.v\n"
        f"export SDC_FILE      = $(DESIGN_HOME)/{platform}/$(DESIGN_NAME)/constraint_{variant}.sdc\n"
        "export CORE_UTILIZATION      ?= 40\n"
        "export PLACE_DENSITY          ?= 0.60\n"
        "export SYNTH_REPEATABLE_BUILD ?= 1\n"
        f"export VERILOG_TOP_PARAMS = LANES {lanes} ACC_W {acc_w}\n"
    )


def _stage_inputs(platform: str, variant: str, lanes: int, acc_w: int, clk_ns: float) -> Path:
    """Copy RTL into the work tree and write the per-variant config.mk + SDC.
    Returns the path to the generated config.mk."""
    cfgdir = MAKE_DIR / platform / DESIGN
    srcdir = MAKE_DIR / "src" / DESIGN
    srcdir.mkdir(parents=True, exist_ok=True)
    cfgdir.mkdir(parents=True, exist_ok=True)

    for f in RTL_FILES:
        shutil.copy(RTL_DIR / f, srcdir / f)

    base_sdc = cfgdir / "constraint.sdc"
    if not base_sdc.exists():
        raise FileNotFoundError(
            f"base SDC missing: {base_sdc} (expected from physical/orfs/make/{platform}/{DESIGN}/)"
        )
    gen_sdc = cfgdir / f"constraint_{variant}.sdc"
    gen_sdc.write_text(
        re.sub(r"(?m)^set clk_period.*$", f"set clk_period    {clk_ns}", base_sdc.read_text())
    )

    gen_cfg = cfgdir / f"config_{variant}.mk"
    gen_cfg.write_text(_config_mk(platform, variant, lanes, acc_w))
    return gen_cfg


# ── Public entry point ────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def run_physical(lanes: int, acc_w: int, clk_ns: float, platform: str = "nangate45") -> dict:
    """Run the full RTL→GDS flow for one config and return parsed metrics.

    Deterministic for a fixed RTL + PDK, so memoised.  Reuses an already-built
    variant (skips the flow if its 6_final.gds exists).

    Returns a dict with: lanes, acc_w, clk_ns, platform, variant, status,
    area_um2, util_pct, wns_ns, tns_ns, setup_viol, power_mw, fmax_mhz,
    period_min_ns, timing_met, gds, report.
    """
    variant = variant_name(lanes, acc_w, clk_ns)
    base = {"lanes": lanes, "acc_w": acc_w, "clk_ns": float(clk_ns),
            "platform": platform, "variant": variant}

    if os.environ.get("PHYSICAL_MOCK"):
        return {**base, "status": "mock", **_mock_metrics(lanes, acc_w, clk_ns)}

    env_sh = ORFS_DIR / "env.sh"
    if not env_sh.exists():
        raise FileNotFoundError(
            f"ORFS not found at {ORFS_DIR} (set ORFS_DIR, or PHYSICAL_MOCK=1 to test offline)"
        )

    gen_cfg = _stage_inputs(platform, variant, lanes, acc_w, clk_ns)
    gds = MAKE_DIR / "results" / platform / DESIGN / variant / "6_final.gds"

    status = "ok"
    if not gds.exists():
        make_cmd = (
            f"source '{env_sh}' && "
            f"make --file='{ORFS_DIR}/flow/Makefile' "
            f"FLOW_HOME='{ORFS_DIR}/flow' WORK_HOME='{MAKE_DIR}' "
            f"DESIGN_CONFIG='{gen_cfg}' FLOW_VARIANT='{variant}'"
        )
        proc = subprocess.run(
            ["bash", "-c", make_cmd], cwd=str(MAKE_DIR),
            capture_output=True, text=True, timeout=ORFS_TIMEOUT,
        )
        # Persist the flow output so a failure is debuggable (mirrors sweep.sh's
        # per-variant log), rather than discarding captured stdout/stderr.
        (MAKE_DIR / f"opt_{variant}.log").write_text(
            (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or "")
        )
        if proc.returncode != 0:
            status = "FAIL"

    metrics = _parse_metrics(MAKE_DIR, platform, variant, clk_ns)
    if status == "ok" and metrics.get("report") is None:
        status = "FAIL"          # flow claimed success but produced no report
    return {**base, "status": status, **metrics}


# ── Fast proxy: Yosys synthesis + OpenROAD STA (no place & route) ──────────────

def _yosys_synth_script(lanes: int, acc_w: int, lib: Path, netlist: Path) -> str:
    # Written to a .ys file (not passed via -p): yosys's command tokenizer does
    # NOT strip shell quotes, so paths are left UNQUOTED — safe here as no path
    # in this flow contains spaces.  Same chparam mechanism ORFS uses.
    rtl = " ".join(str(RTL_DIR / f) for f in RTL_FILES)
    return "\n".join([
        f"read_verilog -sv {rtl}",
        f"chparam -set LANES {lanes} {DESIGN}",
        f"chparam -set ACC_W {acc_w} {DESIGN}",
        f"synth -top {DESIGN} -flatten",
        f"dfflibmap -liberty {lib}",
        f"abc -liberty {lib}",
        "opt_clean -purge",
        f"stat -liberty {lib}",
        f"write_verilog -noattr {netlist}",
        "",
    ])


def _sta_script(lefs: list[Path], lib: Path, netlist: Path, sdc: Path) -> str:
    # Pre-layout STA in OpenROAD: cell delays only (optimistic, no net RC), but
    # fast and — unlike the routed flow — target-clock-independent, so Fmax is a
    # fair cross-config speed metric.  OpenROAD's link_design needs a technology,
    # so read the tech + std-cell LEF first.  report_clock_min_period prints the
    # same "period_min = X fmax = Y" line the full-flow parser already handles.
    lef_lines = "".join(f"read_lef {lef}\n" for lef in lefs)
    return (
        lef_lines
        + f"read_liberty {lib}\n"
        + f"read_verilog {netlist}\n"
        + f"link_design {DESIGN}\n"
        + f"read_sdc {sdc}\n"
        + "report_clock_min_period\n"
        + "report_wns\n"
        + "report_tns\n"
    )


@lru_cache(maxsize=None)
def run_synth_sta(lanes: int, acc_w: int, clk_ns: float, platform: str = "nangate45") -> dict:
    """Fast proxy: synthesise (Yosys) + static-timing (OpenROAD), no P&R.

    Seconds instead of minutes.  Returns the SAME metric shape as run_physical
    (area_um2, fmax_mhz, wns_ns, tns_ns, timing_met, …) so the reward/env are
    unchanged — area is the synth cell area scaled by the placement-inflation
    factor to approximate die area; timing is pre-layout (optimistic).
    """
    variant = variant_name(lanes, acc_w, clk_ns)
    base = {"lanes": lanes, "acc_w": acc_w, "clk_ns": float(clk_ns),
            "platform": platform, "variant": variant}

    if os.environ.get("PHYSICAL_MOCK"):
        return {**base, "status": "mock-proxy", **_mock_metrics(lanes, acc_w, clk_ns)}

    env_sh = ORFS_DIR / "env.sh"
    if not env_sh.exists():
        raise FileNotFoundError(
            f"ORFS not found at {ORFS_DIR} (set ORFS_DIR, or PHYSICAL_MOCK=1 to test offline)"
        )
    lib_rel = _LIBERTY.get(platform)
    if lib_rel is None or platform not in _LEF:
        raise ValueError(f"no proxy lib/lef for platform '{platform}' (try nangate45 / sky130hd)")
    lib  = ORFS_DIR / "flow" / lib_rel
    lefs = [ORFS_DIR / "flow" / rel for rel in _LEF[platform]]

    gen_cfg = _stage_inputs(platform, variant, lanes, acc_w, clk_ns)   # also writes the SDC
    sdc = gen_cfg.parent / f"constraint_{variant}.sdc"

    work = MAKE_DIR / "proxy" / variant
    work.mkdir(parents=True, exist_ok=True)
    netlist = work / "netlist.v"
    synth_ys = work / "synth.ys"
    sta_tcl = work / "sta.tcl"
    synth_ys.write_text(_yosys_synth_script(lanes, acc_w, lib, netlist))
    sta_tcl.write_text(_sta_script(lefs, lib, netlist, sdc))

    # 1) synthesis (script FILE, not -p, to avoid shell-quoting issues; no -q,
    #    which would suppress the `stat` output we parse).
    p1 = subprocess.run(
        ["bash", "-c", f"source '{env_sh}' && yosys '{synth_ys}'"],
        cwd=str(MAKE_DIR), capture_output=True, text=True, timeout=PROXY_TIMEOUT,
    )
    (work / "synth.log").write_text((p1.stdout or "") + "\n--- stderr ---\n" + (p1.stderr or ""))
    if p1.returncode != 0 or not netlist.exists():
        return {**base, "status": "FAIL", "stage": "synth",
                "area_um2": None, "fmax_mhz": None, "wns_ns": None, "tns_ns": None,
                "timing_met": None, "power_mw": None}

    # take the LAST stat (after opt_clean), not intermediate synth/abc stats
    cells_all = re.findall(r"Number of cells:\s+(\d+)", p1.stdout)
    chip_all  = re.findall(r"Chip area for module.*?:\s+([\d.]+)", p1.stdout)
    cell_count = int(cells_all[-1]) if cells_all else None
    cell_area  = float(chip_all[-1]) if chip_all else None
    est_area   = round(cell_area * _PLACE_INFLATION, 1) if cell_area else None

    # 2) static timing (OpenROAD, pre-layout)
    p2 = subprocess.run(
        ["bash", "-c", f"source '{env_sh}' && openroad -no_init -exit '{sta_tcl}'"],
        cwd=str(MAKE_DIR), capture_output=True, text=True, timeout=PROXY_TIMEOUT,
    )
    sta_out = (p2.stdout or "")
    (work / "sta.log").write_text(sta_out + "\n--- stderr ---\n" + (p2.stderr or ""))

    fmax = period_min = None
    m = re.search(r"period_min\s*=\s*([\d.]+).*?fmax\s*=\s*([\d.]+)", sta_out)
    if m:
        period_min = float(m.group(1))
        fmax = float(m.group(2))
    wns = _last_float_on_line(sta_out, "wns")
    tns = _last_float_on_line(sta_out, "tns")
    # Fallback fmax from slack if report_clock_min_period was unavailable.
    if fmax is None and wns is not None and clk_ns - wns > 0:
        period_min = round(clk_ns - wns, 3)
        fmax = round(1000.0 / period_min, 2)

    timing_met = (wns >= 0.0) if wns is not None else None

    return {
        **base, "status": "ok", "stage": "proxy",
        "cells": cell_count,
        "cell_area_um2": cell_area,
        "area_um2": est_area,            # die-area estimate (cell × inflation)
        "util_pct": None,
        "wns_ns": wns, "tns_ns": tns,
        "fmax_mhz": fmax, "period_min_ns": period_min,
        "setup_viol": None, "power_mw": None,
        "timing_met": timing_met,
        "gds": None, "report": str(work / "sta.log"),
    }


# ── Mock mode (no OpenROAD) ───────────────────────────────────────────────────

def _mock_metrics(lanes: int, acc_w: int, clk_ns: float) -> dict:
    """Plausible synthetic metrics for offline testing of the optimizer loop.
    Anchored to the real nangate45 LANES=4 numbers; area grows sub-linearly with
    lanes (per the measured synth sweep), Fmax ~constant (requantize-limited)."""
    # synth-sweep-shaped cell area, inflated to ~P&R scale (×1.35)
    cell_area = {1: 12331, 2: 13120, 4: 14589, 8: 17354, 16: 22949}.get(lanes, 14589)
    area = round(cell_area * 1.353, 1)                      # → ~19738 at lanes=4
    fmax = 269.0                                            # requantize-limited, lane-independent
    period_min = round(1000.0 / fmax, 2)
    met = clk_ns >= period_min
    wns = round(clk_ns - 3.82, 3)                           # crit path ≈ 3.82 ns
    power = round(900.0 + 30.0 * lanes, 1)                  # rough lane scaling, mW
    return {
        "area_um2": area, "util_pct": 47.0,
        "wns_ns": wns, "tns_ns": round(min(wns, 0.0) * 15, 2),
        "setup_viol": 0 if met else 40,
        "power_mw": power, "fmax_mhz": fmax, "period_min_ns": period_min,
        "timing_met": met,
        "gds": None, "report": "mock",
    }


if __name__ == "__main__":
    import json
    import sys
    # Usage: physical_runner.py [--proxy] LANES ACC_W CLK_NS [PLATFORM]
    #   --proxy → fast synth+STA; otherwise full RTL→GDS.
    argv = sys.argv[1:]
    proxy = "--proxy" in argv
    argv = [a for a in argv if a != "--proxy"]
    L = int(argv[0]) if len(argv) > 0 else 4
    A = int(argv[1]) if len(argv) > 1 else 24
    C = float(argv[2]) if len(argv) > 2 else 5.0
    P = argv[3] if len(argv) > 3 else "nangate45"
    fn = run_synth_sta if proxy else run_physical
    print(json.dumps(fn(L, A, C, P), indent=2))

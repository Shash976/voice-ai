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
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    A = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    C = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    print(json.dumps(run_physical(L, A, C), indent=2))

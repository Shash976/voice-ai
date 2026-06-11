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

import hashlib
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

from recipe import (
    RECIPES,
    config_mk_lines,
    recipe_suffix,
    resolve_recipe,
    write_abc_constr,
    yosys_abc_line,
)

_REPO    = Path(__file__).resolve().parent.parent
MAKE_DIR = _REPO / "physical" / "orfs" / "make"
RTL_DIR  = _REPO / "rtl" / "accel"
DESIGN   = "tinymac_accel"
RTL_FILES = ("tinymac_accel.v", "int8_mac_array.v", "requantize.v")

# V1: single source of truth for asap7 vs nangate45 time units.
# asap7 SDCs are in picoseconds; nangate45 in nanoseconds.
# Multiply the optimizer's ns clock by this factor when writing the SDC.
# Divide parsed report time values by this factor before storing in *_ns keys.
PLATFORM_TIME_UNIT: dict[str, float] = {"nangate45": 1.0, "asap7": 1000.0}

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

# V3: RTL content hash — computed once per process (lazy cache).
# Encodes the concatenated contents of the three RTL files (sorted by name)
# so any RTL edit produces a new 8-hex digest → old cached results dirs become
# naturally unreachable without explicit invalidation.
_rtl_hash_cache: str | None = None


def _rtl_hash() -> str:
    """Return an 8-hex-digit sha256 digest of the three RTL source files."""
    global _rtl_hash_cache
    if _rtl_hash_cache is None:
        h = hashlib.sha256()
        for fname in sorted(RTL_FILES):          # sorted for determinism
            p = RTL_DIR / fname
            try:
                h.update(p.read_bytes())
            except OSError:
                h.update(fname.encode())         # file missing: use name as placeholder
        _rtl_hash_cache = h.hexdigest()[:8]
    return _rtl_hash_cache


# Explicit result cache replacing lru_cache.
# V10: only cache results whose status is "ok" or "mock" — never cache TIMEOUT,
# FAIL, or PARSE_FAIL so a transient build failure doesn't poison future calls.
_physical_cache: dict[tuple, dict] = {}
_synth_sta_cache: dict[tuple, dict] = {}


def variant_name(lanes: int, acc_w: int, clk_ns: float,
                 util: int = 40, density: float = 0.60, abc: str | None = None,
                 abc_recipe: str | None = None) -> str:
    """Compute a unique, deterministic variant identifier.

    abc_recipe wins over legacy abc kwarg (resolve_recipe handles both).
    The recipe suffix is appended before the RTL hash, in the slot formerly
    occupied by the raw '_area' suffix (V8 fix extended to all recipes):
        orfs_speed → no suffix      (default, matches pre-recipe naming)
        orfs_area  → "_area"
        plain      → "_plain"
    """
    # V11: use {:.4g} so 1.25 and 1.2 produce distinct tokens ("1p25" vs "1p2");
    # the old :.1f caused variant_name(4,24,1.25)==variant_name(4,24,1.2).
    clk_str = f"{float(clk_ns):.4g}".replace(".", "p")
    name = f"L{lanes}_A{acc_w}_c{clk_str}"
    # Flow-param suffixes are appended ONLY when non-default, so configs that use
    # the original util=40/density=0.60 keep the exact old variant name and reuse
    # any GDS already built by run.sh / sweep.sh.
    if util != 40:
        name += f"_u{util}"
    if abs(float(density) - 0.60) > 1e-9:
        name += f"_d{f'{float(density):.2f}'.replace('.', 'p')}"
    # Stage-B recipe axis (replaces the raw V8 abc=='area' suffix check):
    # resolve_recipe() handles legacy abc='speed'/'area' and new abc_recipe= kwarg.
    resolved = resolve_recipe(abc_recipe=abc_recipe, abc=abc)
    name += recipe_suffix(resolved)
    # V3: append RTL content hash so any RTL edit produces a new variant name
    # and old built results dirs become naturally unreachable.
    name += f"_r{_rtl_hash()}"
    return name


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

    # V1: divide all time-valued report fields by the platform time unit so that
    # stored *_ns keys are always in nanoseconds regardless of the platform's
    # native SDC/report unit (asap7 uses picoseconds → divide by 1000).
    time_div = PLATFORM_TIME_UNIT.get(platform, 1.0)

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
    # Divide raw values by time_div to normalise to nanoseconds.
    raw_wns = _last_float_on_line(rpt_txt, "wns")
    raw_tns = _last_float_on_line(rpt_txt, "tns")
    out["wns_ns"] = (raw_wns / time_div) if raw_wns is not None else None
    out["tns_ns"] = (raw_tns / time_div) if raw_tns is not None else None

    # Fmax + min period: "core_clock period_min = 3.72 fmax = 268.64"
    # period_min is in the platform's native unit; divide to get ns.
    # fmax is in MHz regardless of unit (it's derived as 1/period by ORFS),
    # but for asap7 reported fmax is 1000/period_ps = MHz directly, so no
    # conversion needed for fmax_mhz.
    m = re.search(r"period_min\s*=\s*([\d.]+).*?fmax\s*=\s*([\d.]+)", rpt_txt)
    if m:
        out["period_min_ns"] = float(m.group(1)) / time_div
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

    # V2: if we couldn't parse the essential metrics, flag as PARSE_FAIL so the
    # reward function never silently substitutes reference values for missing data.
    # Required: area_um2 must be present, AND at least one of fmax_mhz / wns_ns.
    if out["area_um2"] is None or (out["fmax_mhz"] is None and out["wns_ns"] is None):
        out["status"] = "PARSE_FAIL"
    else:
        out["status"] = "ok"

    return out


# ── ORFS config generation (mirrors sweep.sh) ─────────────────────────────────

def _config_mk(platform: str, variant: str, lanes: int, acc_w: int,
               util: int = 40, density: float = 0.60, abc: str | None = None,
               abc_recipe: str | None = None) -> str:
    """Generate the per-variant ORFS config.mk content.

    The abc recipe is written as ABC_AREA=1 for orfs_area (the only ORFS
    mechanism); orfs_speed uses the default (omit the variable).
    config_mk_lines() from recipe.py raises ValueError for 'plain' (not
    reachable in the full flow without patching ORFS — use the proxy instead).
    """
    resolved = resolve_recipe(abc_recipe=abc_recipe, abc=abc)
    # config_mk_lines raises ValueError for 'plain' — let it propagate so
    # run_physical callers see a clear error message.
    abc_lines = "\n".join(config_mk_lines(resolved))
    if abc_lines:
        abc_lines += "\n"
    return (
        "export DESIGN_HOME = .\n"
        f"export DESIGN_NAME = {DESIGN}\n"
        f"export PLATFORM    = {platform}\n"
        "export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/tinymac_accel.v \\\n"
        "                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/int8_mac_array.v \\\n"
        "                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/requantize.v\n"
        f"export SDC_FILE      = $(DESIGN_HOME)/{platform}/$(DESIGN_NAME)/constraint_{variant}.sdc\n"
        f"export CORE_UTILIZATION      = {util}\n"
        f"export PLACE_DENSITY          = {density}\n"
        f"{abc_lines}"
        "export SYNTH_REPEATABLE_BUILD = 1\n"
        f"export VERILOG_TOP_PARAMS = LANES {lanes} ACC_W {acc_w}\n"
    )


def _stage_inputs(platform: str, variant: str, lanes: int, acc_w: int, clk_ns: float,
                  util: int = 40, density: float = 0.60, abc: str | None = None,
                  abc_recipe: str | None = None) -> Path:
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
    # V1: multiply the optimizer's ns clock by the platform time unit factor so
    # the SDC gets the value in the platform's native unit (ps for asap7).
    time_unit = PLATFORM_TIME_UNIT.get(platform, 1.0)
    sdc_clk_value = clk_ns * time_unit
    gen_sdc = cfgdir / f"constraint_{variant}.sdc"
    gen_sdc.write_text(
        re.sub(r"(?m)^set clk_period.*$", f"set clk_period    {sdc_clk_value}", base_sdc.read_text())
    )

    gen_cfg = cfgdir / f"config_{variant}.mk"
    gen_cfg.write_text(_config_mk(platform, variant, lanes, acc_w, util, density, abc, abc_recipe))
    return gen_cfg


# ── Public entry point ────────────────────────────────────────────────────────

def run_physical(lanes: int, acc_w: int, clk_ns: float, platform: str = "nangate45",
                 util: int = 40, density: float = 0.60,
                 abc: str | None = None, abc_recipe: str | None = None) -> dict:
    """Run the full RTL→GDS flow for one config and return parsed metrics.

    Deterministic for a fixed RTL + PDK + flow params, so cached in an explicit
    dict.  V10: only successful ("ok") results are cached; TIMEOUT/FAIL/PARSE_FAIL
    results are NOT cached so a transient failure can be retried.  Reuses an
    already-built variant (skips the flow if its 6_final.gds exists).

    abc_recipe (Stage-B): canonical recipe name — {orfs_speed, orfs_area, plain}.
    abc (legacy): accepts 'speed'/'area'; mapped via resolve_recipe().
    abc_recipe wins when both are given.
    'plain' is proxy-only and raises ValueError here (cannot be reproduced
    through ORFS without patching synth_preamble.tcl).

    util / density / abc_recipe are ORFS flow knobs (CORE_UTILIZATION,
    PLACE_DENSITY, ABC_AREA); at their defaults (40, 0.60, orfs_speed) the
    variant name is identical to run.sh/sweep.sh so previously-built GDSs
    are reused.

    Returns a dict with: lanes, acc_w, clk_ns, platform, variant, status,
    area_um2, util_pct, wns_ns, tns_ns, setup_viol, power_mw, fmax_mhz,
    period_min_ns, timing_met, gds, report.
    """
    resolved_recipe = resolve_recipe(abc_recipe=abc_recipe, abc=abc)
    if resolved_recipe == "plain":
        raise ValueError(
            "run_physical does not support abc_recipe='plain'. "
            "The 'plain' recipe (bare abc -liberty) cannot be reproduced in the "
            "ORFS full flow without patching synth_preamble.tcl. "
            "Use run_synth_sta(..., abc_recipe='plain') for the F2 proxy instead."
        )

    cache_key = (lanes, acc_w, float(clk_ns), platform, util, float(density), resolved_recipe)
    if cache_key in _physical_cache:
        return _physical_cache[cache_key]

    variant = variant_name(lanes, acc_w, clk_ns, util, density, abc_recipe=resolved_recipe)
    base = {"lanes": lanes, "acc_w": acc_w, "clk_ns": float(clk_ns),
            "platform": platform, "variant": variant,
            "core_utilization": util, "place_density": density,
            "abc": abc, "abc_recipe": resolved_recipe}

    if os.environ.get("PHYSICAL_MOCK"):
        result = {**base, "status": "mock", **_mock_metrics(lanes, acc_w, clk_ns)}
        _physical_cache[cache_key] = result
        return result

    env_sh = ORFS_DIR / "env.sh"
    if not env_sh.exists():
        raise FileNotFoundError(
            f"ORFS not found at {ORFS_DIR} (set ORFS_DIR, or PHYSICAL_MOCK=1 to test offline)"
        )

    gen_cfg = _stage_inputs(platform, variant, lanes, acc_w, clk_ns, util, density,
                            abc_recipe=resolved_recipe)
    gds = MAKE_DIR / "results" / platform / DESIGN / variant / "6_final.gds"

    flow_status = "ok"
    if not gds.exists():
        make_cmd = (
            f"source '{env_sh}' && "
            f"make --file='{ORFS_DIR}/flow/Makefile' "
            f"FLOW_HOME='{ORFS_DIR}/flow' WORK_HOME='{MAKE_DIR}' "
            f"DESIGN_CONFIG='{gen_cfg}' FLOW_VARIANT='{variant}'"
        )
        # V10: use Popen + communicate(timeout) + start_new_session so we can
        # kill the entire process group on TimeoutExpired (not just the bash wrapper).
        proc = subprocess.Popen(
            ["bash", "-c", make_cmd], cwd=str(MAKE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=ORFS_TIMEOUT)
        except subprocess.TimeoutExpired:
            # Kill the entire process group (yosys/openroad children included).
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            log_path = MAKE_DIR / f"opt_{variant}.log"
            log_path.write_text(f"[TIMEOUT after {ORFS_TIMEOUT}s]\n")
            # V10: return TIMEOUT status (never raise); do NOT cache so the config
            # can be retried and the agent can update on it.
            return {**base, "status": "TIMEOUT",
                    "area_um2": None, "util_pct": None, "wns_ns": None,
                    "tns_ns": None, "setup_viol": None, "power_mw": None,
                    "fmax_mhz": None, "period_min_ns": None,
                    "timing_met": None, "gds": None, "report": str(log_path)}

        # Persist the flow output so a failure is debuggable.
        (MAKE_DIR / f"opt_{variant}.log").write_text(
            (stdout or "") + "\n--- stderr ---\n" + (stderr or "")
        )
        if proc.returncode != 0:
            flow_status = "FAIL"

    metrics = _parse_metrics(MAKE_DIR, platform, variant, clk_ns)
    # V2: _parse_metrics now sets status="PARSE_FAIL" if essential metrics are
    # missing; honour that over the flow_status, and also catch the case where
    # the report file is absent.
    parse_status = metrics.pop("status", "ok")
    if flow_status == "ok" and metrics.get("report") is None:
        flow_status = "FAIL"
    final_status = parse_status if parse_status != "ok" else flow_status

    result = {**base, "status": final_status, **metrics}
    # V10: only cache successful results.
    if final_status == "ok":
        _physical_cache[cache_key] = result
    return result


# ── Gate 2: RTL elaboration check (Yosys hierarchy, ~1-2 s) ────────────────────

_elaborate_cache: dict[tuple, dict] = {}


def run_elaborate(lanes: int, acc_w: int, platform: str = "nangate45") -> dict:
    """Cheap structural gate: does the parameterised RTL elaborate cleanly?

    Reads the Verilog, applies the LANES/ACC_W chparams (exactly as ORFS will),
    builds the hierarchy and runs Yosys `check`. Catches parameter combinations
    that don't elaborate BEFORE paying for synthesis/STA/P&R. Seconds, not minutes.

    Returns {"ok": bool, "stage": "elaborate", "log": <path or note>}.
    """
    cache_key = (lanes, acc_w, platform)
    if cache_key in _elaborate_cache:
        return _elaborate_cache[cache_key]

    if os.environ.get("PHYSICAL_MOCK"):
        # The current RTL elaborates for any positive LANES/ACC_W; model that.
        ok = lanes >= 1 and acc_w >= 1
        result = {"ok": ok, "stage": "elaborate", "log": "mock"}
        _elaborate_cache[cache_key] = result
        return result

    env_sh = ORFS_DIR / "env.sh"
    if not env_sh.exists():
        raise FileNotFoundError(
            f"ORFS not found at {ORFS_DIR} (set ORFS_DIR, or PHYSICAL_MOCK=1 to test offline)"
        )

    rtl = " ".join(str(RTL_DIR / f) for f in RTL_FILES)
    work = MAKE_DIR / "elaborate" / variant_name(lanes, acc_w, 1.0)
    work.mkdir(parents=True, exist_ok=True)
    ys = work / "elaborate.ys"
    ys.write_text("\n".join([
        f"read_verilog -sv {rtl}",
        f"chparam -set LANES {lanes} {DESIGN}",
        f"chparam -set ACC_W {acc_w} {DESIGN}",
        f"hierarchy -check -top {DESIGN}",
        "proc",
        "check -assert",
        "",
    ]))
    p = subprocess.run(
        ["bash", "-c", f"source '{env_sh}' && yosys '{ys}'"],
        cwd=str(MAKE_DIR), capture_output=True, text=True, timeout=PROXY_TIMEOUT,
    )
    (work / "elaborate.log").write_text((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
    result = {"ok": p.returncode == 0, "stage": "elaborate", "log": str(work / "elaborate.log")}
    # Always cache elaborate results (elaboration is deterministic for given RTL hash)
    _elaborate_cache[cache_key] = result
    return result


# ── Fast proxy: Yosys synthesis + OpenROAD STA (no place & route) ──────────────

def _yosys_synth_script(lanes: int, acc_w: int, lib: Path, netlist: Path,
                        abc_recipe: str = "orfs_speed", platform: str = "nangate45",
                        clk_ns: float | None = None, work_dir: Path | None = None) -> str:
    """Return a Yosys synthesis script string.

    Stage-B recipe extension: the abc command is now recipe-aware.
      plain      → bare `abc -liberty <lib>`   (current/legacy proxy behaviour)
      orfs_speed → `abc -script abc_speed.script -liberty ... -constr ... [-D ...]`
      orfs_area  → `abc -script abc_area.script -liberty ... -constr ... [-D ...]`

    For orfs_speed/area, a abc.constr file is written to work_dir (or a temp
    location) so that -constr resolves to an actual file.  This replicates what
    ORFS's synth_preamble.tcl writes to $OBJECTS_DIR/abc.constr (BUF_X1, 3.898 fF
    for nangate45), ensuring F2 proxy and F3 full flow see identical cell
    constraints.

    Written to a .ys file (not passed via -p): yosys's command tokenizer does
    NOT strip shell quotes, so paths are left UNQUOTED — safe here as no path
    in this flow contains spaces.  Same chparam mechanism ORFS uses.
    """
    resolved = resolve_recipe(abc_recipe)
    rtl = " ".join(str(RTL_DIR / f) for f in RTL_FILES)

    if resolved == "plain":
        abc_cmd = f"abc -liberty {lib}"
    else:
        # Write the abc.constr file so -constr has a real target.
        constr_dir = work_dir if work_dir is not None else lib.parent
        constr_path = write_abc_constr(constr_dir, platform)
        abc_cmd = yosys_abc_line(resolved, platform, clk_ns, lib,
                                 constr_path=constr_path)

    return "\n".join([
        f"read_verilog -sv {rtl}",
        f"chparam -set LANES {lanes} {DESIGN}",
        f"chparam -set ACC_W {acc_w} {DESIGN}",
        f"synth -top {DESIGN} -flatten",
        f"dfflibmap -liberty {lib}",
        abc_cmd,
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


def run_synth_sta(lanes: int, acc_w: int, clk_ns: float, platform: str = "nangate45",
                  abc: str | None = None, abc_recipe: str | None = None) -> dict:
    """Fast proxy: synthesise (Yosys) + static-timing (OpenROAD), no P&R.

    Stage-B recipe extension:
      abc_recipe (new): canonical recipe {orfs_speed, orfs_area, plain}.
      abc (legacy):  'speed'/'area' aliases; abc_recipe wins when both given.
      Default: orfs_speed (ORFS default — same script as the full flow uses).

    For orfs_speed/area the proxy uses the same ORFS abc_{speed,area}.script
    with the same -constr file as the full flow (F2 and F3 share recipes),
    so the proxy's cell area distribution matches the full-flow's, improving
    the ρ correlation.  'plain' keeps the legacy bare `abc -liberty` behaviour
    (useful for calibration; not reachable in F3).

    Seconds instead of minutes.  Returns the SAME metric shape as run_physical
    (area_um2, fmax_mhz, wns_ns, tns_ns, timing_met, …) so the reward/env are
    unchanged — area is the synth cell area scaled by the placement-inflation
    factor to approximate die area; timing is pre-layout (optimistic).
    """
    resolved_recipe = resolve_recipe(abc_recipe=abc_recipe, abc=abc)
    cache_key = (lanes, acc_w, float(clk_ns), platform, resolved_recipe)
    if cache_key in _synth_sta_cache:
        return _synth_sta_cache[cache_key]

    variant = variant_name(lanes, acc_w, clk_ns, abc_recipe=resolved_recipe)
    base = {"lanes": lanes, "acc_w": acc_w, "clk_ns": float(clk_ns),
            "platform": platform, "variant": variant,
            "abc": abc, "abc_recipe": resolved_recipe}

    if os.environ.get("PHYSICAL_MOCK"):
        result = {**base, "status": "mock-proxy", **_mock_metrics(lanes, acc_w, clk_ns)}
        _synth_sta_cache[cache_key] = result
        return result

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
    # Pass recipe, platform, clk_ns and work_dir so the constr file is written
    # inside the proxy work directory (same isolation as the full flow).
    synth_ys.write_text(_yosys_synth_script(lanes, acc_w, lib, netlist,
                                            abc_recipe=resolved_recipe,
                                            platform=platform, clk_ns=clk_ns,
                                            work_dir=work))
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

    result = {
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
    # V10: only cache successful proxy results
    if result["status"] == "ok":
        _synth_sta_cache[cache_key] = result
    return result


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

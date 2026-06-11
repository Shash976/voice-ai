"""optimizer/recipe.py — ABC synthesis recipe axis for the TinyMAC design-space.

**Rationale** (from docs/07_rl_pipeline_design.md Phase 4 P1 / Phase 5 Exp 4):
The ABC recipe is the highest-variance synthesis lever on this design:
  plain        ≈ 14.4 K µm²  (bare abc -liberty, no script, no constr)
  orfs_area    ≈ 17.2 K µm²  (ORFS abc_area.script + -constr/-D)
  orfs_speed   ≈ 20.7 K µm²  (ORFS abc_speed.script + -constr/-D)
43% spread in synthesis area; the proxy currently always synthesises with
*plain* while the full flow always uses *orfs_speed* — this mismatch degrades
ρ below what it should be (Exp 3 ρ=0.9; expected to improve once both fidelities
share a recipe).

**ORFS mechanism** (verified against /opt/OpenROAD-flow-scripts):
  synth_preamble.tcl lines 140-146:
      if { $::env(ABC_AREA) } {
          set abc_script $SCRIPTS_DIR/abc_area.script
      } else {
          set abc_script $SCRIPTS_DIR/abc_speed.script
      }
  abc_args = [-script <script> <lib_args> -constr abc.constr]  (line 164)
  -D <clock_period> appended when SDC_FILE_CLOCK_PERIOD is set  (line 173)
  ABC_AREA default = 0  (variables.json "default": 0)

  Two-value mapping:
      orfs_speed  → ABC_AREA=0 (or omit — default is speed)
      orfs_area   → ABC_AREA=1

**'plain' at F3 (full flow)**:
  ORFS does NOT expose a variable to override the abc script path — it only
  selects between abc_speed.script and abc_area.script via ABC_AREA (0/1).
  There is no SYNTH_ARGS path that skips the script selection in
  synth_preamble.tcl.  Patching /opt is forbidden.
  → `run_physical(..., abc_recipe="plain")` raises ValueError.
  'plain' is proxy-only (F2): the current proxy uses bare `abc -liberty` which
  is exactly the 'plain' recipe.  This is useful for calibrating the proxy's
  bias against F3 and for exploring the synthesis frontier cheaply.

**variant_name integration** (V8 fix extended):
  orfs_speed → no suffix       (matches the no-suffix default; GDS reuse)
  orfs_area  → "_area" suffix
  plain      → "_plain" suffix
  The recipe suffix is placed before the RTL hash (same slot as the old _area).

**Backward compatibility** (legacy abc= kwarg):
  Existing callers pass abc="speed" or abc="area" (physical_runner.py V8 era).
  The resolve_recipe() helper maps these to the new names so physical_runner.py
  need not be patched at call sites.
"""

from __future__ import annotations

from pathlib import Path

# ── Public recipe set ────────────────────────────────────────────────────────

# Ordered: proxy-only first so callers can gate on index >= 0.
RECIPES: tuple[str, ...] = ("orfs_speed", "orfs_area", "plain")

# ORFS scripts root — used to build absolute paths in yosys -script invocations.
ORFS_SCRIPTS_DIR: Path = Path(
    __import__("os").environ.get("ORFS_DIR", "/opt/OpenROAD-flow-scripts")
) / "flow" / "scripts"

# ABC driver/load for nangate45 (from platforms/nangate45/config.mk).
# The proxy replicates the constr file that synth_preamble.tcl writes to
# OBJECTS_DIR/abc.constr so both fidelities use the same physical constraints.
_ABC_CONSTR: dict[str, tuple[str, float]] = {
    "nangate45": ("BUF_X1", 3.898),
    # sky130hd would go here when wired; placeholder to avoid KeyError
    "sky130hd":  ("sky130_fd_sc_hd__buf_1", 5.0),
}


def resolve_recipe(abc_recipe: str | None = None, abc: str | None = None) -> str:
    """Resolve abc_recipe and legacy abc= kwargs to a canonical recipe name.

    abc_recipe wins if both are given.  Legacy mappings:
        'speed'  → 'orfs_speed'
        'area'   → 'orfs_area'
        None     → 'orfs_speed'   (ORFS default)

    Raises ValueError for unrecognised names.
    """
    # abc_recipe wins over legacy abc
    raw = abc_recipe if abc_recipe is not None else abc

    # None → ORFS default (speed script)
    if raw is None:
        return "orfs_speed"

    # Legacy aliases
    _LEGACY = {"speed": "orfs_speed", "area": "orfs_area"}
    canonical = _LEGACY.get(raw, raw)

    if canonical not in RECIPES:
        raise ValueError(
            f"Unknown abc recipe {raw!r}. "
            f"Valid values: {RECIPES} (legacy aliases: 'speed', 'area')"
        )
    return canonical


def recipe_suffix(recipe: str) -> str:
    """Return the variant_name suffix for the given recipe.

    orfs_speed → ""        (default: no suffix, matches the pre-recipe era)
    orfs_area  → "_area"
    plain      → "_plain"
    """
    recipe = resolve_recipe(recipe)
    return {"orfs_speed": "", "orfs_area": "_area", "plain": "_plain"}[recipe]


def yosys_abc_args(recipe: str, platform: str, clk_ns: float | None,
                   lib: str | Path) -> list[str]:
    """Return the yosys `abc` command arguments for a given recipe.

    Parameters
    ----------
    recipe:   canonical recipe name (one of RECIPES) or legacy alias.
    platform: PDK platform key (e.g. 'nangate45').
    clk_ns:   clock period in ns used for the -D delay target argument.
              If None or ≤ 0, -D is omitted (safe for area-mode; the
              ORFS speed script uses -D for delay-targeted mapping but
              P1 shows -D is a no-op in yosys 0.64 regardless).
    lib:      path to the standard-cell liberty file (passed as -liberty
              and inside the -constr file).

    Returns
    -------
    List of arguments to append after `abc` in a Yosys script, e.g.:
        ["abc", "-script", "/path/abc_speed.script",
         "-liberty", "/path/NangateOpenCellLibrary_typical.lib",
         "-constr", "/path/abc.constr", "-D", "4000.0"]
    For 'plain', returns the minimal:
        ["abc", "-liberty", "/path/..."]

    The constr file is written separately by the caller (see
    `write_abc_constr`).  This function only emits the argument list.
    """
    recipe = resolve_recipe(recipe)
    lib = str(lib)

    if recipe == "plain":
        # Bare abc: no script, no constr, no -D — exactly the current proxy.
        return ["-liberty", lib]

    script_name = "abc_speed.script" if recipe == "orfs_speed" else "abc_area.script"
    script_path = str(ORFS_SCRIPTS_DIR / script_name)

    args = ["-script", script_path, "-liberty", lib]

    # -constr placeholder; the actual file path is injected by the caller after
    # write_abc_constr() places it.  We use the sentinel "<CONSTR>" so callers
    # can str.replace or pass write_abc_constr's return value.
    # We follow the same pattern as synth_preamble.tcl line 164-165.
    args += ["-constr", "<CONSTR>"]

    # -D delay target: period in picoseconds (ABC internal unit is ps).
    # Only append when meaningful.
    if clk_ns is not None and clk_ns > 0:
        args += ["-D", str(round(clk_ns * 1000.0, 1))]

    return args


def write_abc_constr(work_dir: Path, platform: str) -> Path:
    """Write the abc.constr file (driving-cell + load) for the given platform.

    This replicates what ORFS synth_preamble.tcl writes to
    $OBJECTS_DIR/abc.constr (lines 178-181).  The proxy must create this file
    and pass its path as the -constr argument to abc.

    Returns the Path of the written file.
    """
    driver, load = _ABC_CONSTR.get(platform, ("BUF_X1", 3.898))
    constr_path = work_dir / "abc.constr"
    constr_path.write_text(
        f"set_driving_cell {driver}\n"
        f"set_load {load}\n"
    )
    return constr_path


def yosys_abc_line(recipe: str, platform: str, clk_ns: float | None,
                   lib: str | Path, constr_path: str | Path | None = None) -> str:
    """Return a complete yosys `abc ...` command string for embedding in a .ys script.

    Parameters
    ----------
    constr_path: Path to the abc.constr file written by write_abc_constr().
                 Required for orfs_speed / orfs_area; ignored for plain.
                 If None for a non-plain recipe, "<CONSTR>" sentinel is used.
    """
    args = yosys_abc_args(recipe, platform, clk_ns, lib)
    if constr_path is not None:
        args = [str(constr_path) if a == "<CONSTR>" else a for a in args]
    return "abc " + " ".join(args)


def config_mk_lines(recipe: str) -> list[str]:
    """Return the config.mk export lines needed for the given recipe.

    Used by physical_runner._config_mk to emit the ABC_AREA flag when
    building the full ORFS flow config.mk for a variant.

    orfs_speed → []                            (ABC_AREA defaults to 0)
    orfs_area  → ["export ABC_AREA = 1"]
    plain      → raises ValueError             (not reachable via ORFS)
    """
    recipe = resolve_recipe(recipe)
    if recipe == "plain":
        raise ValueError(
            "The 'plain' recipe (bare abc -liberty) cannot be reproduced through "
            "the ORFS full flow without patching synth_preamble.tcl. "
            "Use 'plain' only with run_synth_sta (F2 proxy). "
            "For the full flow, choose 'orfs_speed' or 'orfs_area'."
        )
    if recipe == "orfs_area":
        return ["export ABC_AREA = 1"]
    # orfs_speed: omit ABC_AREA (default = 0 = speed)
    return []

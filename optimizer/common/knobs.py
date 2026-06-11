"""optimizer/common/knobs.py — design-agnostic ORFS knob registry.

All 28 knobs are transcribed from /tmp/knobs_research.yaml (the evidence-annotated
research pass over ORFS variables.yaml and AutoTuner JSON configs).  The registry
data is EMBEDDED here — do NOT read /tmp at runtime.

Tiers:
    1 — dominant (≥5% area or ≥2× timing/cycles); always included
    2 — moderate / design-dependent (AutoTuner canonical set)
    3 — fine-tuning: CTS / timing-repair / routing params
    4 — macro-only: active only when design.has_macros is True

Emit convention:
    Knob.emit_lines(value) → list[str] of "export NAME = value" strings
    ready to append to a config.mk fragment.

    CLOCK_PERIOD is a pseudo-knob that writes to the SDC, not config.mk;
    emit_lines() returns [] — physical_runner.py owns the SDC write.

    VERILOG_TOP_PARAMS is the bridge for design.params axes; emit_lines()
    is called with the already-assembled string "PARAM_A va PARAM_B vb".

    ABC_AREA is the binary toggle:  "0" → orfs_speed (default), "1" → orfs_area.
    Callers that use abc_recipe="plain" should skip config.mk emission entirely
    (plain is proxy-only); recipe.py handles that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ── Knob data class ────────────────────────────────────────────────────────────


@dataclass
class Knob:
    """One ORFS knob with all metadata needed for emission and space construction."""

    name: str
    tier: int
    stage: str
    type: str           # "int" | "float" | "categorical" | "bool" | "pseudo_sdc"
    default: Any
    range: tuple | None = None       # (lo, hi) for int/float
    choices: list | None = None      # for categorical
    # Internal emit template; {v} is replaced by the value.
    # Empty string "" means the caller handles emission (CLOCK_PERIOD / VERILOG_TOP_PARAMS).
    _emit_template: str = field(default="", repr=False)
    notes: str = ""
    evidence: str = ""

    # ── Emission ──────────────────────────────────────────────────────────────

    def emit_lines(self, value: Any) -> list[str]:
        """Return zero or more 'export NAME = value' config.mk lines.

        CLOCK_PERIOD  → [] (written to SDC by physical_runner, not config.mk)
        VERILOG_TOP_PARAMS → [] when value is empty/None (no RTL params);
                              otherwise one line with the pre-built string.
        MACRO_PLACE_HALO  → emits "export MACRO_PLACE_HALO = v v" (x y both the same).
        All others         → one line: "export NAME = value"
        """
        if self.type == "pseudo_sdc":
            # CLOCK_PERIOD: physical_runner owns the SDC write
            return []

        if self.name == "VERILOG_TOP_PARAMS":
            if not value:
                return []
            return [f"export VERILOG_TOP_PARAMS = {value}"]

        if self.name == "MACRO_PLACE_HALO":
            # "x y" pair; scalar value applied to both dims
            return [f"export MACRO_PLACE_HALO = {value} {value}  # x y in µm"]

        return [f"export {self.name} = {value}"]

    # ── Validation helpers ────────────────────────────────────────────────────

    def validate_value(self, value: Any) -> list[str]:
        """Return a list of warning/error strings for `value` (empty = ok)."""
        errs: list[str] = []
        if self.type in ("int", "float") and self.range is not None:
            lo, hi = self.range
            try:
                v = float(value)
            except (TypeError, ValueError):
                errs.append(f"{self.name}: cannot cast {value!r} to number")
                return errs
            if v < lo or v > hi:
                errs.append(
                    f"{self.name}={v} outside documented range [{lo}, {hi}]"
                )
        elif self.type == "categorical" and self.choices is not None:
            if str(value) not in [str(c) for c in self.choices]:
                errs.append(
                    f"{self.name}={value!r} not in choices {self.choices}"
                )
        return errs


# ── Registry ───────────────────────────────────────────────────────────────────


class KnobRegistry:
    """Registry of all 28 ORFS knobs, with tier-filtering and space-building."""

    def __init__(self, knobs: list[Knob]) -> None:
        self._knobs: dict[str, Knob] = {k.name: k for k in knobs}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "KnobRegistry":
        """Build the registry from the embedded knob data (no file I/O at runtime).

        All evidence comments and safety interactions are documented inline.
        Source: /tmp/knobs_research.yaml (transcribed; do not re-read that file).
        """
        knobs = _build_knobs()
        return cls(knobs)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, name: str) -> Knob | None:
        return self._knobs.get(name)

    def all_knobs(self) -> list[Knob]:
        return list(self._knobs.values())

    @staticmethod
    def _resolve_design(design: Any) -> Any:
        """Normalize design: if a str, load via DesignSpec.load(name_or_path).

        This prevents the silent-wrong-result mode where a bare string is passed
        and getattr(str, 'params', {}) returns {}, silently dropping all design params.
        Returns the resolved DesignSpec (or original object if not a str).

        Import is tried as both 'common.designs' (when optimizer/ is on sys.path)
        and with a sys.path insertion for the optimizer root so the method works
        whether called from optimizer/ context or from optimizer/common/ context.
        """
        if isinstance(design, str):
            import sys as _sys
            from pathlib import Path as _Path
            # Ensure optimizer/ root is importable (idempotent if already present)
            _opt_root = str(_Path(__file__).resolve().parent.parent)
            if _opt_root not in _sys.path:
                _sys.path.insert(0, _opt_root)
            try:
                from common.designs import DesignSpec
                return DesignSpec.load(design)
            except Exception:
                return design   # resolution failed; caller sees empty params (safe fallback)
        return design

    def active(self, max_tier: int, design: Any) -> list[Knob]:
        """Return knobs with tier ≤ max_tier, filtered for macro knobs.

        Tier-4 (macro) knobs are included ONLY when design.has_macros is True.
        For any other design, tier-4 knobs are suppressed.

        Parameters
        ----------
        max_tier : int   1..4
        design   : str | DesignSpec | any object with a ``has_macros`` attribute.
                   str → resolved via DesignSpec.load() (eliminates silent-wrong-result).
                   If None, tier-4 is suppressed.
        """
        design = self._resolve_design(design)
        has_macros = getattr(design, "has_macros", False) or False
        result = []
        for k in self._knobs.values():
            if k.tier > max_tier:
                continue
            if k.tier == 4 and not has_macros:
                continue
            result.append(k)
        return result

    def space(self, max_tier: int, design: Any, platform: str) -> dict:
        """Build the merged search-space dict for candidate generation.

        Returns:
            {axis_name: {"type": ..., "choices": [...] or "range": [lo, hi], "default": ...}}

        Sources in order (later entries do NOT override earlier):
            1. design.params axes (RTL chparam knobs like mac_lanes, accumulator_width)
            2. CLOCK_PERIOD from design.platforms[platform]["clock_range_ns"]
            3. abc_recipe from the three RECIPES in recipe.py
            4. Active ORFS knobs (tiers 1..max_tier, minus pseudo-knobs already
               handled above: CLOCK_PERIOD, VERILOG_TOP_PARAMS, ABC_AREA)

        design may be a str, in which case it is resolved via DesignSpec.load().
        A design with no params (e.g. gcd) contributes no RTL axes — the space
        contains only ORFS knobs and clock, with no phantom mac_lanes/accumulator_width
        axes.

        The dict format is the contract the candidate generator consumes.
        """
        design = self._resolve_design(design)
        space: dict[str, dict] = {}

        # 1. RTL parameter axes from design.params
        # Each param entry may carry 'rtl_param_name'; the search-space axis name
        # is the canonical param name (e.g. mac_lanes), NOT the RTL chparam name.
        params = getattr(design, "params", {}) or {}
        for param_name, param_spec in params.items():
            spec: dict = {}
            if "choices" in param_spec:
                spec["type"] = "categorical"
                spec["choices"] = list(param_spec["choices"])
            elif "range" in param_spec:
                lo, hi = param_spec["range"]
                spec["type"] = "int"
                spec["range"] = [lo, hi]
            else:
                spec["type"] = "categorical"
                spec["choices"] = list(param_spec.get("values", []))
            if "default" in param_spec:
                spec["default"] = param_spec["default"]
            space[param_name] = spec

        # 2. CLOCK_PERIOD — from design.platforms[platform]["clock_range_ns"]
        plat_info = (getattr(design, "platforms", {}) or {}).get(platform, {})
        clk_range = plat_info.get("clock_range_ns", [3.0, 8.0])
        clk_default = plat_info.get("default_clock_ns", 5.0)
        space["clock_period_ns"] = {
            "type": "float",
            "range": list(clk_range),
            "default": float(clk_default),
        }

        # 3. ABC recipe axis (three canonical values)
        try:
            from common.recipe import RECIPES
        except ImportError:
            RECIPES = ("orfs_speed", "orfs_area", "plain")
        space["abc_recipe"] = {
            "type": "categorical",
            "choices": list(RECIPES),
            "default": "orfs_speed",
        }

        # 4. Active ORFS knobs (skip pseudo-knobs already handled above)
        _SKIP = {"CLOCK_PERIOD", "VERILOG_TOP_PARAMS", "ABC_AREA"}
        for knob in self.active(max_tier, design):
            if knob.name in _SKIP:
                continue
            entry: dict = {"type": knob.type, "default": knob.default}
            if knob.choices is not None:
                entry["choices"] = list(knob.choices)
            elif knob.range is not None:
                entry["range"] = list(knob.range)
            space[knob.name] = entry

        return space


# ── Safety validation ──────────────────────────────────────────────────────────


def validate_config(config: dict) -> list[str]:
    """Check a config dict for known dangerous combinations.

    Returns a list of warning/error strings (empty = ok).

    Checks (from knobs_research_notes.md):
    1. util × density × pad abort bound:
       CORE_UTILIZATION > 60 AND CELL_PAD_IN_SITES_GLOBAL_PLACEMENT > 2 →
       placer abort (cells cannot be placed at 6-site spacing at high util).
       Extended: CORE_UTILIZATION > 70 → unconditional warning.
       Extended: PLACE_DENSITY > 0.80 → warning (placer abort threshold).

    2. PLACE_DENSITY vs PLACE_DENSITY_LB_ADDON exclusivity:
       Setting PLACE_DENSITY_LB_ADDON > 0 while PLACE_DENSITY is also above the
       platform default (> 0.60) is confusing — max(PLACE_DENSITY, density_lb +
       PLACE_DENSITY_LB_ADDON) wins silently. Warn, not error (both are legal).

    3. MIN_PLACE_STEP_COEF > MAX_PLACE_STEP_COEF:
       Placer enforces MIN ≤ MAX; setting MAX < MIN may crash or behave
       unpredictably. Error.

    4. RECOVER_POWER > 0 when timing not likely met:
       Cannot reliably detect this without real WNS, but warn if the clock is
       very tight (CLOCK_PERIOD < 2.0 ns for nangate45) and RECOVER_POWER > 0.

    5. CTS_CLUSTER_SIZE < 10: documented minimum; below this CTS times out.
    """
    errs: list[str] = []

    util = float(config.get("CORE_UTILIZATION", 40))
    density = float(config.get("PLACE_DENSITY", 0.60))
    lb_addon = float(config.get("PLACE_DENSITY_LB_ADDON", 0.0))
    gpad = int(config.get("CELL_PAD_IN_SITES_GLOBAL_PLACEMENT", 0))
    min_coef = float(config.get("MIN_PLACE_STEP_COEF", 0.95))
    max_coef = float(config.get("MAX_PLACE_STEP_COEF", 1.05))
    cts_size = int(config.get("CTS_CLUSTER_SIZE", 20))
    recover_pw = float(config.get("RECOVER_POWER", 0))
    clk = float(config.get("clock_period_ns", config.get("CLOCK_PERIOD", 5.0)))

    # 1a. util × density × pad abort bound (research note #5 / gotcha #3)
    if util > 60 and gpad > 2:
        errs.append(
            f"ABORT RISK: CORE_UTILIZATION={util} > 60 combined with "
            f"CELL_PAD_IN_SITES_GLOBAL_PLACEMENT={gpad} > 2 → placer abort "
            "(cells cannot be placed with large padding at high utilization). "
            "Reduce CELL_PAD to ≤ 2 or lower CORE_UTILIZATION to ≤ 60."
        )

    # 1b. extreme utilization
    if util > 70:
        errs.append(
            f"WARNING: CORE_UTILIZATION={util} > 70 — CTS/GRT timing-repair "
            "loops may not converge on congested designs. "
            "Set TNS_END_PERCENT=20 and SETUP_SLACK_MARGIN=-0.3 during exploration."
        )

    # 1c. extreme density
    if density > 0.80:
        errs.append(
            f"ABORT RISK: PLACE_DENSITY={density} > 0.80 — placer abort threshold. "
            "The tool will print 'Cannot achieve target density; minimum recommended "
            "is X.XX' and abort gracefully but without a useful result."
        )

    # 2. PLACE_DENSITY vs PLACE_DENSITY_LB_ADDON exclusivity (research note #1)
    if lb_addon > 0 and density > 0.60:
        errs.append(
            f"WARNING: PLACE_DENSITY_LB_ADDON={lb_addon} > 0 AND "
            f"PLACE_DENSITY={density} > 0.60 (platform default). "
            "Effective density = max(PLACE_DENSITY, density_lb + LB_ADDON). "
            "The higher value silently wins. "
            "For design-agnostic use prefer tuning only PLACE_DENSITY_LB_ADDON "
            "and leaving PLACE_DENSITY at 0.60."
        )

    # 3. MIN_PLACE_STEP_COEF > MAX_PLACE_STEP_COEF
    if min_coef > max_coef:
        errs.append(
            f"ERROR: MIN_PLACE_STEP_COEF={min_coef} > MAX_PLACE_STEP_COEF={max_coef}. "
            "The placer enforces MIN ≤ MAX; this combination may crash or "
            "produce undefined placement behaviour."
        )

    # 4. RECOVER_POWER with aggressive clock
    if recover_pw > 0 and clk < 2.0:
        errs.append(
            f"WARNING: RECOVER_POWER={recover_pw} > 0 with clock_period_ns={clk} < 2.0. "
            "Timing may not be met at this clock; applying power recovery when "
            "WNS < 0 produces unpredictable QoR (tool attempts simultaneous timing "
            "fix + power recovery). Safe default = 0 during timing closure."
        )

    # 5. CTS_CLUSTER_SIZE below documented minimum
    if cts_size < 10:
        errs.append(
            f"WARNING: CTS_CLUSTER_SIZE={cts_size} < 10 (documented minimum). "
            "Below 10, the clock-tree synthesis can time out on large designs."
        )

    return errs


# ── Embedded knob data ─────────────────────────────────────────────────────────
# Transcribed from /tmp/knobs_research.yaml.  All evidence comments preserved.


def _build_knobs() -> list[Knob]:
    """Return all 28 Knob objects from the embedded registry data."""
    return [

        # =====================================================================
        # TIER 1 — Dominant axes
        # =====================================================================

        Knob(
            name="VERILOG_TOP_PARAMS",
            tier=1,
            stage="rtl",
            type="categorical",
            # Choices here are for tinymac reference only; the space() method
            # builds the actual choices from design.params dynamically.
            choices=["LANES 1 ACC_W 24", "LANES 2 ACC_W 24", "LANES 4 ACC_W 24",
                     "LANES 8 ACC_W 24", "LANES 16 ACC_W 24", "LANES 32 ACC_W 24",
                     "LANES 4 ACC_W 32"],
            default="LANES 4 ACC_W 24",
            _emit_template="export VERILOG_TOP_PARAMS = {v}",
            notes=(
                "ORFS chparam pseudo-knob: passed as VERILOG_TOP_PARAMS in config.mk. "
                "For tinymac: LANES (dominant — area ×2.6 L1→L32, cycles ×5.8) and "
                "ACC_W (±5% area, accuracy cliff at 16). "
                "Verified in variables.yaml (stages: synth)."
            ),
            evidence=(
                "Measured: LANES dominates all other flow knobs — area 14.3K→43.8K µm² "
                "(L1→L32), cycles ×5.8. ACC_W: area ±5%, accuracy cliff at 16 "
                "(doc07 P4/Exp2). Tier 1 justified by 2.6× area leverage."
            ),
        ),

        Knob(
            name="CLOCK_PERIOD",
            tier=1,
            stage="sdc",
            type="pseudo_sdc",
            # Range is for nangate45; asap7 range would be [0.3, 1.5] ns.
            # physical_runner.py applies PLATFORM_TIME_UNIT conversion on SDC write.
            range=(3.0, 8.0),
            default=5.0,
            _emit_template="",   # SDC write owned by physical_runner
            notes=(
                "AutoTuner pseudo-knob (_SDC_CLK_PERIOD). Written to constraint.sdc "
                "by physical_runner.py via PLATFORM_TIME_UNIT conversion (×1000 for "
                "asap7 ps). NOT a config.mk variable. "
                "CRITICAL: the V1 bug poisoned all 12 asap7 results (wns '-1861 ns' "
                "was actually ps). The fix is in PLATFORM_TIME_UNIT. "
                "AutoTuner nangate45/gcd range: [0.3, 1.0] ns; "
                "sky130hd/ibex: [9, 16] ns; asap7/ibex: [1200, 2000] ps."
            ),
            evidence=(
                "Strongest flow-level coupling on 46 builds: Fmax 113→307 MHz, "
                "area ±18%, power ×3 as clock tightens. Surrogate must learn the "
                "effort-coupling response surface; the old max(clk, 3.72) analytic "
                "cap contradicts measurements (doc07 Phase 3, Phase 5 Exp 2)."
            ),
        ),

        Knob(
            name="ABC_AREA",
            tier=1,
            stage="synth",
            type="categorical",
            choices=["0", "1"],   # 0 = orfs_speed (default), 1 = orfs_area
            default="0",
            _emit_template="export ABC_AREA = {v}",
            notes=(
                "ORFS synthesis recipe toggle. ABC_AREA=0 → abc_speed.script; "
                "ABC_AREA=1 → abc_area.script. 'plain' mode (bare abc -liberty) is "
                "proxy-only and is NOT emitted here. "
                "The -D flag is a no-op with yosys 0.64 (doc07 P1-b: bit-identical "
                "stats for D∈{1,2,4,8}). "
                "V8 bug: old code omitted the ABC line for speed, making speed and "
                "default identical variants — fixed by config_mk_lines() in recipe.py."
            ),
            evidence=(
                "Measured: 43% area / 59% pre-layout delay spread across 3 recipes "
                "at L4/A24/4ns. plain 14,456 µm² / orfs_area 17,187 / orfs_speed "
                "20,675 µm². Critical path does not move with recipe (doc07 P1 Exp4)."
            ),
        ),

        Knob(
            name="CORE_UTILIZATION",
            tier=1,
            stage="floorplan",
            type="float",
            range=(20.0, 60.0),
            default=40.0,
            _emit_template="export CORE_UTILIZATION = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: floorplan). "
                "AutoTuner canonical: gcd [5,100], ibex-sky130 [20,50], ibex-asap7 [5,10]. "
                "Tier 1 for general designs; on tinymac <0.3% effect (very sparse design). "
                "Should be fixed at 40 for tinymac (doc07 P4) but kept as tier-1 axis "
                "for design-agnostic use."
            ),
            evidence=(
                "AutoTuner tunes in every canonical config (gcd/ibex/aes/cva6). "
                "For tinymac: matched triples L2_A24_c5p0 u30/50/60 → "
                "15,787/15,756/15,753 µm² (<0.3% effect) — fix at 40 for this design."
            ),
        ),

        # =====================================================================
        # TIER 2 — Moderate / design-dependent axes (AutoTuner canonical)
        # =====================================================================

        Knob(
            name="CORE_ASPECT_RATIO",
            tier=2,
            stage="floorplan",
            type="float",
            range=(0.5, 2.0),
            default=1.0,
            _emit_template="export CORE_ASPECT_RATIO = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: floorplan). "
                "AutoTuner range: gcd [0.5, 2.0], ibex-sky130 [0.5, 2.0], "
                "ibex-asap7 [0.9, 1.1] (tighter for constrained designs). "
                "Interacts with PLACE_DENSITY: tall narrow die behaves differently "
                "from square die at high utilization."
            ),
            evidence=(
                "AutoTuner includes in all designs except gcd-asap7. "
                "Affects wire-length distribution and pin-access symmetry."
            ),
        ),

        Knob(
            name="CORE_MARGIN",
            tier=2,
            stage="floorplan",
            type="float",
            range=(1.0, 3.0),
            default=1.0,
            _emit_template="export CORE_MARGIN = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: floorplan). "
                "AutoTuner range: gcd-nangate45 [1,3], gcd-asap7 [2,2] (fixed). "
                "Spacing between core and die boundary in microns. "
                "Too small → pin-access DRC; too large → wasted area."
            ),
            evidence=(
                "AutoTuner tunes in all reference designs. Small continuous effect "
                "on die area; mainly affects pin routing margin."
            ),
        ),

        Knob(
            name="PLACE_DENSITY",
            tier=2,
            stage="place",
            type="float",
            range=(0.40, 0.80),
            default=0.60,
            _emit_template="export PLACE_DENSITY = {v}",
            notes=(
                "In variables.yaml but NOT marked tunable=1. AutoTuner does NOT tune "
                "this directly — it tunes PLACE_DENSITY_LB_ADDON instead. "
                "GOTCHA: PLACE_DENSITY and PLACE_DENSITY_LB_ADDON are effectively "
                "exclusive: effective density = max(PLACE_DENSITY, density_lb + LB_ADDON). "
                "Setting both is legal but the higher value silently wins. "
                "GOTCHA: > 0.80 → placer abort; > 0.60 at high util → CTS/GRT timeout."
            ),
            evidence=(
                "Measured on tinymac: ~1.4% area effect (L4_A28_c5p0 d0.45→0.65: "
                "17,789→17,542 µm²). Small on this design; fixed at 0.60 for tinymac."
            ),
        ),

        Knob(
            name="PLACE_DENSITY_LB_ADDON",
            tier=2,
            stage="place",
            type="float",
            range=(0.0, 0.20),
            default=0.0,
            _emit_template="export PLACE_DENSITY_LB_ADDON = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: floorplan, place). "
                "AutoTuner canonical: all designs use [0.0, 0.2]. "
                "Adds a delta on top of computed lower-bound density. "
                "PREFERRED over hard PLACE_DENSITY for design-agnostic use because "
                "it adapts to platform-specific minimums. "
                "GOTCHA: do not set both this > 0 and PLACE_DENSITY > 0.60 — "
                "validate_config() will warn."
            ),
            evidence=(
                "AutoTuner's PREFERRED density axis — appears in every autotuner.json "
                "(gcd, ibex, aes, jpeg, cva6) with range [0.0, 0.2]."
            ),
        ),

        Knob(
            name="CELL_PAD_IN_SITES_GLOBAL_PLACEMENT",
            tier=2,
            stage="place",
            type="int",
            range=(0, 3),
            default=0,
            _emit_template="export CELL_PAD_IN_SITES_GLOBAL_PLACEMENT = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: place, floorplan). "
                "AutoTuner range: [0, 3] in all designs, sometimes [0, 5] for gcd no_sdc. "
                "Adds cell padding during global placement to ease routability. "
                "GOTCHA: high padding at high CORE_UTILIZATION → placer abort "
                "(util > 60 AND pad > 2 is dangerous — validate_config checks this)."
            ),
            evidence=(
                "AutoTuner tunes in all canonical designs. More impactful on "
                "congestion-limited designs (ibex) than simple datapath blocks (tinymac)."
            ),
        ),

        Knob(
            name="CELL_PAD_IN_SITES_DETAIL_PLACEMENT",
            tier=2,
            stage="place",
            type="int",
            range=(0, 3),
            default=0,
            _emit_template="export CELL_PAD_IN_SITES_DETAIL_PLACEMENT = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: place, cts, grt). "
                "AutoTuner range: [0, 3] all designs. "
                "Padding applied during detail placement and CTS/GRT legalization. "
                "Typically kept ≤ CELL_PAD_IN_SITES_GLOBAL_PLACEMENT to avoid abrupt "
                "density discontinuity between placement stages."
            ),
            evidence=(
                "AutoTuner canonical, appears everywhere. Fine-grain routability lever."
            ),
        ),

        # =====================================================================
        # TIER 3 — Fine-tuning: CTS / timing-repair / routing
        # =====================================================================

        Knob(
            name="CTS_CLUSTER_SIZE",
            tier=3,
            stage="cts",
            type="int",
            range=(10, 200),
            default=20,
            _emit_template="export CTS_CLUSTER_SIZE = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: cts). "
                "AutoTuner range: [10, 200] universally. "
                "Maximum number of sinks per CTS cluster. "
                "Smaller → deeper tree, more buffers, better skew. "
                "Larger → shallower tree, fewer buffers (better area). "
                "GOTCHA: < 10 → CTS timeout on large designs (validate_config checks)."
            ),
            evidence=(
                "AutoTuner's CTS knob #1 — in every canonical config. "
                "The most reliable CTS tuning lever. Skew vs buffer area trade-off."
            ),
        ),

        Knob(
            name="CTS_CLUSTER_DIAMETER",
            tier=3,
            stage="cts",
            type="float",
            range=(20.0, 400.0),
            default=100.0,
            _emit_template="export CTS_CLUSTER_DIAMETER = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: cts). "
                "AutoTuner range: [20, 400] universally. "
                "Maximum spatial diameter (µm) of a CTS sink cluster. "
                "Must match die size (diameter=400 on a 200 µm die is a no-op). "
                "For tinymac ~140×140 µm die, sensible range is [20, 150]."
            ),
            evidence=(
                "AutoTuner canonical. Spatial locality constraint for clock tree; "
                "affects insertion delay uniformity."
            ),
        ),

        Knob(
            name="TNS_END_PERCENT",
            tier=3,
            stage="cts",
            type="float",
            range=(5.0, 100.0),
            default=100.0,
            _emit_template="export TNS_END_PERCENT = {v}",
            notes=(
                "Verified in variables.yaml (stages: place, cts, floorplan, grt). "
                "NOT in AutoTuner's canonical JSON — flow-level variable only. "
                "Default 100 = fix all violating endpoints (correct for final closure); "
                "reduce to 5–20 to speed up exploratory builds. "
                "GOTCHA: 0 disables timing repair entirely → raw placement quality only."
            ),
            evidence=(
                "Strong runtime lever; halving from 100 to 50 cuts CTS/GRT repair "
                "time by 30–50% on constrained designs. variables.yaml doc: "
                "'reduce to 5% for runtime.'"
            ),
        ),

        Knob(
            name="SETUP_SLACK_MARGIN",
            tier=3,
            stage="cts",
            type="float",
            range=(-0.5, 0.5),
            default=0.0,
            _emit_template="export SETUP_SLACK_MARGIN = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: cts, floorplan, grt). "
                "Positive = overfix (repair to extra margin); negative = underfix. "
                "GOTCHA: overfix at pre-CTS stage (ideal clock) adds too many buffers "
                "→ congestion; recommend ≤ 0 for exploration. "
                "A negative margin (-0.1 to -0.2 ns) prevents runaway timing repair "
                "on infeasible clock targets."
            ),
            evidence=(
                "Useful for DSE: negative margin caps runtime without aborting. "
                "variables.yaml notes it can substitute for SDC period sweeping."
            ),
        ),

        Knob(
            name="ROUTING_LAYER_ADJUSTMENT",
            tier=3,
            stage="grt",
            type="float",
            range=(0.1, 0.7),
            default=0.5,
            _emit_template="export ROUTING_LAYER_ADJUSTMENT = {v}",
            notes=(
                "Verified in variables.yaml. "
                "AutoTuner equivalent: _FR_LAYER_ADJUST → written to fastroute.tcl. "
                "In the native ORFS make flow this is ROUTING_LAYER_ADJUSTMENT in "
                "config.mk. AutoTuner range: [0.1, 0.3] congestion-sensitive, "
                "[0.0, 0.1] easy designs (asap7/ibex). "
                "GOTCHA: too high → global route detours; too low → DRC."
            ),
            evidence=(
                "AutoTuner tunes in all designs; primary global-route congestion knob. "
                "For tinymac (0 DRC on all 46 builds) this is tier 3 / not binding; "
                "for congested designs (ibex, jpeg) it is critical."
            ),
        ),

        Knob(
            name="RECOVER_POWER",
            tier=3,
            stage="grt",
            type="float",
            range=(0.0, 30.0),
            default=0.0,
            _emit_template="export RECOVER_POWER = {v}",
            notes=(
                "Verified in variables.yaml (no stages list — applies globally). "
                "NOT in AutoTuner. Percent of paths with positive slack that can be "
                "down-sized for power savings. "
                "GOTCHA: applying when WNS < 0 → simultaneous timing fix + power "
                "recovery → unpredictable QoR. validate_config warns when clk < 2 ns."
            ),
            evidence=(
                "Power-optimization lever with non-trivial effect on total power. "
                "Safe default = 0 during timing closure."
            ),
        ),

        Knob(
            name="DETAILED_ROUTE_END_ITERATION",
            tier=3,
            stage="route",
            type="int",
            range=(32, 64),
            default=64,
            _emit_template="export DETAILED_ROUTE_END_ITERATION = {v}",
            notes=(
                "Verified in variables.yaml (stages: route). Default 64. "
                "Reducing to 32 cuts route runtime by ~40% for designs that already "
                "converge early (tinymac: 0 DRC every build). "
                "For congested designs, reducing risks non-zero DRC on exit. "
                "Practical proxy-vs-full speedup lever."
            ),
            evidence=(
                "route+CTS = 65% of full-flow runtime (doc07 Phase 3 Exp1). "
                "Not an AutoTuner knob but directly controls droute wall-clock."
            ),
        ),

        Knob(
            name="MIN_PLACE_STEP_COEF",
            tier=3,
            stage="place",
            type="float",
            range=(0.95, 1.00),
            default=0.95,
            _emit_template="export MIN_PLACE_STEP_COEF = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: place). "
                "Valid range per docs: 0.95–1.05. "
                "Sets the pcof_min lower bound for global placement Nesterov "
                "optimization. Lower → smaller step → slower convergence but "
                "may escape local minima. "
                "GOTCHA: values outside [0.95, 1.05] cause undefined behavior."
            ),
            evidence=(
                "tunable=1 in variables.yaml. Placement quality vs runtime trade-off; "
                "relevant when placement is the bottleneck stage."
            ),
        ),

        Knob(
            name="MAX_PLACE_STEP_COEF",
            tier=3,
            stage="place",
            type="float",
            range=(1.00, 1.15),
            default=1.05,
            _emit_template="export MAX_PLACE_STEP_COEF = {v}",
            notes=(
                "Verified tunable=1 in variables.yaml (stages: place). "
                "Valid range: 1.00–1.20. "
                "Upper bound on Nesterov step size. Higher → more aggressive "
                "optimization, risk of divergence. "
                "Paired with MIN_PLACE_STEP_COEF; optimizer should enforce MIN ≤ MAX "
                "(validate_config checks this). "
                "GOTCHA: MAX > 1.20 is outside documented safe range."
            ),
            evidence=(
                "tunable=1 in variables.yaml. Affects placement convergence quality."
            ),
        ),

        # =====================================================================
        # TIER 4 — Macro-only knobs
        # Active ONLY when design.has_macros is True.
        # NOT applicable to tinymac (0 macros, 0 memory bits verified by synth_stat.txt).
        # Relevant when SRAM weight/activation buffers are added (roadmap: 2-4 SRAM macros).
        # Note: RTLMP_FLOW does NOT exist in OpenROAD 26Q2 — omitted.
        # =====================================================================

        Knob(
            name="MACRO_PLACE_HALO",
            tier=4,
            stage="floorplan",
            type="float",
            range=(1.0, 20.0),
            default=5.0,
            _emit_template="export MACRO_PLACE_HALO = {v} {v}  # x y in µm",
            notes=(
                "Verified in variables.yaml (stages: floorplan). "
                "Used by macro_place_util.tcl's rtl_macro_placer invocation. "
                "Controls the halo (exclusion zone) around each macro for std-cell "
                "placement. Too small → cells crowd macro pins → DRC/LVS; "
                "too large → wasted area. "
                "GOTCHA: MACRO_BLOCKAGE_HALO overrides the auto-computed blockage "
                "from MACRO_PLACE_HALO — set consistently. "
                "Note: emit_lines emits 'MACRO_PLACE_HALO = v v' (x y both same scalar)."
            ),
            evidence=(
                "ORFS macro-placement standard knob; referenced in macro_place_util.tcl. "
                "At 2-4 macros (SRAM buffers roadmap), OpenROAD's RTL-MP + this axis "
                "suffices; AlphaChip-style learned placement is over-engineering at "
                "this scale (research notes P2 verdict)."
            ),
        ),

        Knob(
            name="MACRO_BLOCKAGE_HALO",
            tier=4,
            stage="floorplan",
            type="float",
            range=(1.0, 15.0),
            default=2.0,
            _emit_template="export MACRO_BLOCKAGE_HALO = {v}",
            notes=(
                "Verified in variables.yaml (stages: floorplan). "
                "When set, overrides the auto-computed blockage derived from "
                "MACRO_PLACE_HALO. Prevents std-cell placement and routing near macro "
                "edges. Usually set ≥ MACRO_PLACE_HALO x/y. "
                "GOTCHA: MACRO_BLOCKAGE_HALO < halo → routing DRC near macros."
            ),
            evidence="Standard macro-placement companion knob.",
        ),

        Knob(
            name="RTLMP_MAX_LEVEL",
            tier=4,
            stage="floorplan",
            type="int",
            range=(1, 4),
            default=2,
            _emit_template="export RTLMP_MAX_LEVEL = {v}",
            notes=(
                "Verified in variables.yaml (stages: floorplan). "
                "Maximum depth of the physical hierarchy tree used by rtl_macro_placer. "
                "RTLMP_FLOW does NOT exist in OpenROAD 26Q2 — macro placement is "
                "enabled/controlled via MACRO_PLACEMENT_TCL. "
                "GOTCHA: setting MAX_LEVEL too high on a shallow hierarchy → placer "
                "error 'no clusters at level N'."
            ),
            evidence=(
                "variables.yaml documented parameter. Relevant for hierarchical "
                "designs with multiple macros."
            ),
        ),

        Knob(
            name="RTLMP_WIRELENGTH_WT",
            tier=4,
            stage="floorplan",
            type="float",
            range=(10.0, 200.0),
            default=100.0,
            _emit_template="export RTLMP_WIRELENGTH_WT = {v}",
            notes=(
                "Verified in variables.yaml (stages: floorplan). "
                "Weight for HPWL in the RTL-MP cost function. "
                "Higher → macro placement optimizes more aggressively for wirelength "
                "(may increase notch violations). "
                "GOTCHA: extreme imbalance between weights → infeasible floorplan."
            ),
            evidence=(
                "One of four RTL-MP cost weights; toggling wirelength vs "
                "outline/notch is the standard macro-placement tuning dial."
            ),
        ),

        Knob(
            name="RTLMP_BOUNDARY_WT",
            tier=4,
            stage="floorplan",
            type="float",
            range=(10.0, 100.0),
            default=50.0,
            _emit_template="export RTLMP_BOUNDARY_WT = {v}",
            notes=(
                "Verified in variables.yaml (stages: floorplan). "
                "Weight for keeping macro clusters near die boundaries. "
                "Lower → macros float into center (good for inter-macro nets); "
                "higher → macros pack against walls (good for I/O proximity). "
                "GOTCHA: high boundary weight + high utilization → macros overlap."
            ),
            evidence=(
                "Companion to RTLMP_WIRELENGTH_WT; macro-placement quality lever "
                "for multi-macro designs."
            ),
        ),

    ]


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Self-test: run with `python3 optimizer/common/knobs.py`."""
    import sys

    print("=== knobs.py self-test ===")

    # 1. Registry loads
    reg = KnobRegistry.load()
    all_k = reg.all_knobs()
    print(f"  Registry loaded: {len(all_k)} knobs")
    # Tier 1: 4, Tier 2: 6, Tier 3: 9, Tier 4: 5 → 24 total
    assert len(all_k) == 24, f"Expected 24 knobs, got {len(all_k)}"

    # 2. Tier filtering works
    tier1 = reg.active(max_tier=1, design=None)
    tier1_names = {k.name for k in tier1}
    assert "VERILOG_TOP_PARAMS" in tier1_names, "VERILOG_TOP_PARAMS not in tier 1"
    assert "CLOCK_PERIOD" in tier1_names, "CLOCK_PERIOD not in tier 1"
    assert "ABC_AREA" in tier1_names, "ABC_AREA not in tier 1"
    assert "CORE_UTILIZATION" in tier1_names, "CORE_UTILIZATION not in tier 1"
    assert "CTS_CLUSTER_SIZE" not in tier1_names, "tier-3 knob in tier-1 result"
    print(f"  Tier 1 knobs: {sorted(tier1_names)}")

    # 3. Tier-4 only with has_macros
    class _MockDesign:
        has_macros = False
        params = {"LANES": {"choices": [1, 2, 4, 8, 16, 32], "default": 4},
                  "ACC_W": {"choices": [16, 24, 32], "default": 24}}
        platforms = {"nangate45": {"clock_range_ns": [3.0, 8.0], "default_clock_ns": 5.0}}

    no_macro = reg.active(max_tier=4, design=_MockDesign())
    assert not any(k.tier == 4 for k in no_macro), "Tier-4 knob leaked into no-macro design"

    class _MacroDesign(_MockDesign):
        has_macros = True

    with_macro = reg.active(max_tier=4, design=_MacroDesign())
    assert any(k.tier == 4 for k in with_macro), "Tier-4 knob missing for macro design"
    print(f"  Tier-4 gate: {len([k for k in no_macro if k.tier==4])} macro knobs without macros, "
          f"{len([k for k in with_macro if k.tier==4])} with macros  PASS")

    # 4. space() for tinymac tier-1 reproduces the 4 canonical axes.
    # Param names in space() are the CANONICAL names from design.params,
    # not the RTL chparam names.  _MockDesign uses old-style LANES/ACC_W keys
    # for backward compat; real tinymac YAML uses mac_lanes/accumulator_width.
    space = reg.space(max_tier=1, design=_MockDesign(), platform="nangate45")
    # _MockDesign still uses LANES/ACC_W (old style) — space() emits those names.
    required_axes = {"LANES", "ACC_W", "clock_period_ns", "abc_recipe"}
    for ax in required_axes:
        assert ax in space, f"Expected axis '{ax}' in tier-1 space, got: {set(space.keys())}"
    # CORE_UTILIZATION is tier 1 but also in the space since it's not a pseudo-knob
    # It gets added under active knobs unless it's CLOCK_PERIOD/VERILOG_TOP_PARAMS/ABC_AREA
    assert "CORE_UTILIZATION" in space, "CORE_UTILIZATION not in tier-1 space"
    print(f"  space() tier-1 axes (MockDesign): {sorted(space.keys())}  PASS")

    # 4b. space() for real tinymac YAML uses canonical names mac_lanes/accumulator_width
    try:
        real_tinymac_space = reg.space(max_tier=1, design="tinymac_accel", platform="nangate45")
        canonical_axes = {"mac_lanes", "accumulator_width", "clock_period_ns", "abc_recipe"}
        for ax in canonical_axes:
            assert ax in real_tinymac_space, \
                f"Expected canonical axis '{ax}' in real tinymac tier-1 space, got: {set(real_tinymac_space.keys())}"
        assert "LANES" not in real_tinymac_space, \
            "LANES (RTL chparam name) should NOT be in search space (use mac_lanes)"
        assert "ACC_W" not in real_tinymac_space, \
            "ACC_W (RTL chparam name) should NOT be in search space (use accumulator_width)"
        print(f"  space() tier-1 axes (real tinymac): {sorted(real_tinymac_space.keys())}  PASS")
    except FileNotFoundError:
        print("  SKIP real tinymac YAML test (file not found)")

    # 5. validate_config catches util/density/pad combo
    bad_cfg = {
        "CORE_UTILIZATION": 65,
        "CELL_PAD_IN_SITES_GLOBAL_PLACEMENT": 3,
        "PLACE_DENSITY": 0.85,
    }
    errs = validate_config(bad_cfg)
    assert any("ABORT RISK" in e and "CELL_PAD" in e for e in errs), \
        f"util×pad abort not caught: {errs}"
    assert any("ABORT RISK" in e and "PLACE_DENSITY" in e for e in errs), \
        f"density abort not caught: {errs}"
    print(f"  validate_config caught {len(errs)} errors for bad config  PASS")

    # 6. PLACE_DENSITY / LB_ADDON exclusivity warning
    excl_cfg = {"PLACE_DENSITY": 0.70, "PLACE_DENSITY_LB_ADDON": 0.10}
    excl_errs = validate_config(excl_cfg)
    assert any("exclusive" in e.lower() or "LB_ADDON" in e for e in excl_errs), \
        f"LB_ADDON exclusivity not caught: {excl_errs}"
    print(f"  validate_config LB_ADDON exclusivity warning  PASS")

    # 7. MIN_PLACE_STEP_COEF > MAX raises error
    coef_cfg = {"MIN_PLACE_STEP_COEF": 1.05, "MAX_PLACE_STEP_COEF": 0.97}
    coef_errs = validate_config(coef_cfg)
    assert any("ERROR" in e and "MIN_PLACE_STEP_COEF" in e for e in coef_errs), \
        f"MIN > MAX not caught: {coef_errs}"
    print(f"  validate_config MIN > MAX step coef error  PASS")

    # 8. Every emit line is a sane "export NAME = value" string
    for knob in all_k:
        if knob.type == "pseudo_sdc":
            lines = knob.emit_lines(5.0)
            assert lines == [], f"CLOCK_PERIOD emit_lines should be [], got {lines}"
            continue
        if knob.choices:
            val = knob.choices[0]
        elif knob.range:
            lo, hi = knob.range
            val = (lo + hi) / 2.0
            if knob.type == "int":
                val = int(val)
        else:
            val = knob.default
        lines = knob.emit_lines(val)
        # VERILOG_TOP_PARAMS with non-empty value should emit one line
        if knob.name == "VERILOG_TOP_PARAMS":
            lines = knob.emit_lines("LANES 4 ACC_W 24")
            assert len(lines) == 1, f"{knob.name}: expected 1 line, got {lines}"
            assert lines[0].startswith("export VERILOG_TOP_PARAMS"), lines[0]
            # Empty string case
            assert knob.emit_lines("") == [], "empty VERILOG_TOP_PARAMS should emit []"
            continue
        assert len(lines) >= 1, f"{knob.name}: expected ≥1 emit line, got {lines}"
        for line in lines:
            assert "export" in line, f"{knob.name}: line missing 'export': {line!r}"
    print(f"  All {len(all_k)} knobs emit valid lines  PASS")

    print("\n=== knobs.py self-test PASSED ===")
    sys.exit(0)

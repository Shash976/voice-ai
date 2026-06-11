"""constants.py — single source of truth for the TinyMAC behavioral cycle model.

All constants in this file were MEASURED from the Verilator sim after the V13
saturation-order fix (per-chunk, not per-MAC) landed in sim_main.cpp on 2026-06-10.
The V13 fix aligns the behavioral sim with the RTL FSM and the TB golden model:
the int8_mac_array produces a LANES-wide partial sum (psum) in one S_MAC clock
cycle, and acc_sat = saturate(acc + psum) is latched — once per chunk, not once
per individual product.

Sweep command that produced these numbers:
    python3 optimizer/measure_real.py
    (extended to LANES ∈ {1,2,4,8,16,32}, averaging all 64 inference vectors)

Public contract (other modules import exactly these names):
    SW_BASELINE_CYCLES : int              11196638 (Stage-3 no-accel, 100 MHz)
    AVG_CYCLES         : dict[int, int]   lanes -> measured accel cycles/inference
    behavioral_cycles  : (int) -> float   a + b/lanes refit to AVG_CYCLES
    MAX_SPEEDUP_GRID   : int              for the 45-config space (search_space.yaml)
    MAX_SPEEDUP_FULL   : int              for the cascade/full space

Do NOT hardcode these numbers elsewhere — import from here.
"""

from __future__ import annotations

# ── SW baseline (Stage-3, no accelerator) ────────────────────────────────────
# Measured on the Stage-3 PicoRV32 Verilator sim (100 MHz).
# This is unaffected by the accelerator saturation-order fix.
SW_BASELINE_CYCLES: int = 11_196_638

# ── Measured accelerator cycles per inference ─────────────────────────────────
# Sweep: LANES ∈ {1,2,4,8,16,32}, ACC_W=32 (acc_w does not affect cycle count),
# 64 test vectors each, averaged.  Measured 2026-06-10 after V13 fix.
#
# Cycle model is output-stationary with ACCEL_CH_OVERHEAD=2 (bias load + requant):
#   latency = n_outputs × (ceil(K / LANES) + 2)
#
# Sanity anchors from docs/07_rl_pipeline_design.md (measured on same machine):
#   8 lanes ≈ 61,399  →  measured 61,400  ✓ (within 1 cycle)
#  16 lanes ≈ 46,669  →  measured 46,670  ✓ (within 1 cycle)
AVG_CYCLES: dict[int, int] = {
    1:  273_130,
    2:  152_140,
    4:   91_650,
    8:   61_400,
    16:  46_670,
    32:  39_310,
}

# ── behavioral_cycles(lanes): analytic fit to AVG_CYCLES ─────────────────────
# Form: a + b/lanes  (classic output-stationary latency model)
# Fit via least squares over all 6 lane counts (numpy.linalg.lstsq):
#
#   a = 31_452.5   b = 241_567.0
#
# Residuals (measured - fitted):
#   lanes= 1: +110.5   lanes= 2:  -96.0   lanes= 4: -194.2
#   lanes= 8: -248.4   lanes=16: +119.6   lanes=32: +308.5
#
# Max absolute residual: 309 cycles out of ~39K–273K (< 1%).  The fit is accurate
# enough for the reward proxy (which accepts ±a few percent for the cycle term).
_CYCLE_OVERHEAD = 31_452.5   # formerly 28_000 (old synthetic model)
_CYCLE_MAC_WORK = 241_567.0  # formerly 242_000 (old synthetic model)


def behavioral_cycles(lanes: int) -> float:
    """Return estimated accel cycles per TinyVAD inference for `lanes` MAC lanes.

    Uses the a + b/lanes fit to the measured AVG_CYCLES table.  For exact values
    at a measured lane count, prefer AVG_CYCLES[lanes] directly.
    """
    return _CYCLE_OVERHEAD + _CYCLE_MAC_WORK / max(lanes, 1)


# ── max_speedup derivation ────────────────────────────────────────────────────
# MAX_SPEEDUP_GRID: for the 45-config space (5 lanes × 3 acc × 3 clk).
#
# Derivation (frequency-aware, matches reward.real_speedup):
#   SW_BASELINE_LATENCY_NS = 11_196_638 × 10.0 = 111_966_380 ns
#   Max grid config = lanes=16, acc=24, clk=5ns (crit_path=3.72 < 5ns, so no cap):
#     latency_ns   = 46_670 × 5.0 = 233_350 ns
#     real_speedup = 111_966_380 / 233_350 = 479.82×
#   ceil(479.82 × 1.11) = 533 → round up to next 64-boundary = 576
#   576 / 479.82 = 1.20× headroom  (>10% required, 20% achieved)
#
# The old value was also 576 (derivation used a synthetic 28000+242000/lanes model;
# the new measured model gives a *lower* max speedup, making 576 more conservative).
# Grid optimum is unchanged: {lanes=4, acc=24, clk=5}, reward ≈ 3.995 (was 4.012).
MAX_SPEEDUP_GRID: int = 576

# MAX_SPEEDUP_FULL: for the cascade/full space (lanes up to 32).
# lanes=32, acc=24, clk=5ns: real_speedup = 111_966_380 / (39_310 × 5.0) = 569.66×
# 1024 provides 1024/569.66 = 1.80× headroom.  Set by convention (power of 2).
MAX_SPEEDUP_FULL: int = 1024

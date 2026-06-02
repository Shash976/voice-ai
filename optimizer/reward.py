"""reward.py — proxy computations and multi-objective reward for the TinyMAC optimizer.

Reward formula
--------------
    reward = 2.0  * accuracy
           + 3.0  * log2_norm_speedup        # log2(real_speedup)/log2(max_speedup), ∈ [−1, 1]
           − 0.4  * area_proxy               # 1.0 at (8 lanes, 32b acc)
           − 0.4  * power_proxy              # 1.0 at baseline
           − 3.0  * timing_violation         # 0 or 1
           − 8.0  * perf_floor_penalty       # if real_speedup < min_useful_speedup (10×)
           − 50   * overflow_penalty         # if accumulator too narrow for TinyVAD
           − 50   * (1 − accuracy)           # if sim produces wrong outputs

Frequency-coupled performance (Stage-5 fix, 2026-06-02)
-------------------------------------------------------
The speedup that drives the reward is now a *real-time* speedup, not a raw
cycle-count ratio.  Previously the reward used SW_cycles/accel_cycles, which is
frequency-INDEPENDENT — so a slower clock (20 ns) scored identically to a fast
one while making the chip slower in wall-clock terms, and the optimizer always
picked the slowest clock.  We now convert cycles to nanoseconds:

    effective_clock_ns = max(clock_period_ns, critical_path_ns)
        # you cannot run faster than the critical path; an impossible clock is
        # silently capped — no free reward for requesting clk < crit.
    latency_ns   = avg_cycles * effective_clock_ns
    real_speedup = SW_BASELINE_LATENCY_NS / latency_ns

A faster (smaller-period) clock that still meets timing therefore yields a
higher real_speedup (more reward); a clock faster than the critical path gives
NO additional benefit (capped).  real_speedup is computed in env.step (which
holds both avg_cycles and the config) and passed in via proxy_metrics so the
dashboard's cycle-based "speedup" field is untouched.

Wall-clock sim time is NOT in the reward.  It is not a property of the design
and injecting it makes results non-reproducible across machines.

Design notes
------------
* accumulator_width is now a genuinely simulated parameter (sim_main.cpp clips
  the accumulator at runtime).  The proxy overflow check here is kept only as a
  fast pre-filter for agents that want to skip obvious misses without running
  the sim.

* area_proxy is normalised to 1.0 at the baseline config
  (mac_lanes=8, acc_width=32).  The MAC array is modelled as 80% of area;
  the SRAM-buffer term is a fixed 20% (buffer sizes were removed from the
  active search — the behavioral sim has no buffer model, so optimizing them
  would be guessing performance.  Deferred to Stage-6 ORFS where SRAM area is
  real.  area_proxy still uses .get() with a 1024B default so the 20% buffer
  term stays at its baseline value and area=1.0 at baseline.)
"""

from __future__ import annotations

import math

# TinyVAD worst-case signed accumulator magnitude
# Conv0: K=200 (in_ch=40 × kern=5), max |product| = 128×128 = 16384
# (int8 range is −128..127, but zero-points shift them; 128×128 is conservative)
# Max |acc| = 200 × 16384 = 3,276,800
_TINYVAD_MAX_ACC = 3_276_800

_INT_MAX = {16: 32_767, 24: 8_388_607, 32: 2_147_483_647}

# Baseline config for area/power normalisation
_BASELINE_MAC_PRODUCT = 8 * 32       # mac_lanes=8, acc_width=32
_BASELINE_BUF_BYTES   = 1024 + 1024  # input + weight buffers (fixed; see module docstring)

# SW baseline latency, used to convert real_speedup from cycles → nanoseconds.
# SW_BASELINE_CYCLES (= 11,196,638, see runner.py) was measured on the Stage-3
# no-accel sim, which is assumed to run at 10 ns / 100 MHz.  Latency in ns is
# therefore cycles × 10.0.  Kept here (not imported from runner) so reward.py
# has no sim dependency and the sanity test can import it standalone.
SW_BASELINE_CYCLES      = 11_196_638
SW_BASELINE_CLOCK_NS    = 10.0                              # 100 MHz
SW_BASELINE_LATENCY_NS  = SW_BASELINE_CYCLES * SW_BASELINE_CLOCK_NS


# ── Proxy computations ────────────────────────────────────────────────────────

def area_proxy(config: dict) -> float:
    """
    Chip area relative to baseline config (= 1.0).
    Baseline: mac_lanes=8, acc_width=32, buffers=1024B each.

    Model: MAC array = 80% of area (∝ lanes × acc_width),
           SRAM buffers = 20% (∝ total buffer bytes).
    """
    lanes  = config["mac_lanes"]
    acc_w  = config.get("accumulator_width",  32)
    in_buf = config.get("input_buffer_bytes",  1024)
    wt_buf = config.get("weight_buffer_bytes", 1024)

    mac_frac = (lanes * acc_w) / _BASELINE_MAC_PRODUCT   # 1.0 at baseline
    buf_frac = (in_buf + wt_buf) / _BASELINE_BUF_BYTES    # 1.0 at baseline

    return round(0.8 * mac_frac + 0.2 * buf_frac, 4)      # 1.0 at baseline ✓


def power_proxy(config: dict, area: float | None = None) -> float:
    """
    Dynamic power proxy: area × clock_frequency.
    Normalised to 1.0 at baseline (area=1.0, clock=10 ns → 100 MHz).

    NOTE: power uses the *requested* clock_period_ns, not the effective one.
    A config that requests an impossible (sub-critical-path) clock still pays
    the higher dynamic-power cost of that aggressive target — one more reason a
    timing-violating config never out-scores the same config clocked legally.
    """
    if area is None:
        area = area_proxy(config)
    clock_ns = config.get("clock_period_ns", 10)
    freq_mhz = 1000.0 / clock_ns
    return round(area * freq_mhz / 100.0, 4)  # 1.0 at baseline ✓


def critical_path_ns(config: dict) -> float:
    """
    Estimated combinational critical path of the MAC datapath (ns).

    Critical path model (calibrated for ASAP7 7 nm):
      2.5 ns base adder + 0.15 ns routing per lane + 0.05 ns per acc register byte.

    Examples:
      16 lanes, 32b acc:  2.5 + 2.40 + 0.20 = 5.10 ns
       8 lanes, 32b acc:  2.5 + 1.20 + 0.20 = 3.90 ns

    Shared helper: timing_slack_ns() and the effective-clock computation in
    env.step both derive from this, so the timing model is single-sourced.
    """
    lanes = config["mac_lanes"]
    acc_w = config.get("accumulator_width", 32)
    return 2.5 + 0.15 * lanes + 0.05 * (acc_w / 8)


def timing_slack_ns(config: dict) -> float:
    """
    Timing slack = clock_period − critical_path  (ns).
    Positive → timing met; negative → violation (chip won't work at this clock).

    At 16 lanes, 32b acc, 5 ns clock: slack = 5 − 5.10 = −0.1 ns  ← violation
    At  8 lanes, 32b acc, 5 ns clock: slack = 5 − 3.90 =  1.1 ns  ← just OK
    At  8 lanes, 32b acc, 10 ns clock: slack = 10 − 3.90 = 6.1 ns ← comfortable
    """
    clock_ns = config.get("clock_period_ns", 10)
    return round(clock_ns - critical_path_ns(config), 3)


def effective_clock_ns(config: dict) -> float:
    """
    The clock period the silicon actually runs at.

    If you request a clock faster than the critical path you do NOT get it for
    free — the design runs at the critical-path period instead.  This caps the
    benefit of an impossible clock and is what couples real-time performance to
    frequency without rewarding timing violations.
    """
    clock_ns = config.get("clock_period_ns", 10)
    return max(float(clock_ns), critical_path_ns(config))


def real_speedup(config: dict, avg_cycles: float) -> float:
    """
    Frequency-aware (real-time) speedup over the Stage-3 SW baseline.

        latency_ns   = avg_cycles * effective_clock_ns(config)
        real_speedup = SW_BASELINE_LATENCY_NS / latency_ns

    Unlike the cycle-based speedup (SW_cycles/accel_cycles) this rewards a
    faster clock and is capped at the critical-path clock.  This is the value
    the reward's speedup term consumes.
    """
    latency_ns = max(avg_cycles, 1.0) * effective_clock_ns(config)
    return SW_BASELINE_LATENCY_NS / max(latency_ns, 1e-9)


def acc_overflows(config: dict) -> bool:
    """
    Fast proxy check: True if acc_width is analytically too narrow for TinyVAD.
    The sim will also catch this (accuracy < 1.0), but this lets agents skip
    obviously bad configs without launching a subprocess.
    """
    acc_w = config.get("accumulator_width", 32)
    return _TINYVAD_MAX_ACC > _INT_MAX.get(acc_w, _INT_MAX[32])


def compute_proxies(config: dict) -> dict:
    """Return all proxy metrics for a config dict (no sim required).

    real_speedup / latency_ns are NOT computed here because they need
    avg_cycles from the sim.  env.step computes them and merges them into the
    proxy_metrics dict before calling compute_reward (see env.step).
    """
    a      = area_proxy(config)
    p      = power_proxy(config, a)
    slack  = timing_slack_ns(config)
    crit   = critical_path_ns(config)
    return {
        "area_proxy":       a,
        "power_proxy":      p,
        "critical_path_ns": round(crit, 3),
        "effective_clock_ns": round(effective_clock_ns(config), 3),
        "timing_slack_ns":  slack,
        "timing_violation": slack < 0.0,
        "acc_overflow":     acc_overflows(config),
    }


# ── Reward scalar ─────────────────────────────────────────────────────────────

def compute_reward(
    sim_metrics:   dict,
    proxy_metrics: dict,
    weights:       dict | None = None,
) -> float:
    """
    Multi-objective reward (higher is better).  Signature deliberately omits
    elapsed_s — wall-clock time is not a design property and injecting it makes
    rewards non-reproducible across machines and load conditions.

    Parameters
    ----------
    sim_metrics   : dict returned by runner.run_sim()  (accuracy, speedup, …)
    proxy_metrics : dict returned by compute_proxies(), with real_speedup /
                    latency_ns merged in by env.step (area, power, slack, …)
    weights       : optional override dict (keys match search_space.yaml reward section)

    Speedup term
    ------------
    Uses proxy_metrics["real_speedup"] (frequency-aware, ns-based) — NOT the
    cycle-based sim_metrics["speedup"], which is frequency-independent and was
    the root cause of the slow-clock degeneracy.  Falls back to the cycle-based
    value only if real_speedup is absent (e.g. an old record), so the dashboard
    and any legacy caller keep working.
    """
    w = weights or {}
    w_acc    = w.get("w_accuracy",          2.0)
    w_spd    = w.get("w_speedup",           3.0)
    w_area   = w.get("w_area",             -0.4)
    w_pwr    = w.get("w_power",            -0.4)
    w_tv     = w.get("w_timing_violation", -3.0)
    max_spd  = w.get("max_speedup",        576.0)
    min_spd  = w.get("min_useful_speedup",  10.0)
    perf_pen = w.get("perf_floor_penalty",  -8.0)

    accuracy = sim_metrics["accuracy"]
    # Prefer the frequency-aware real-time speedup; fall back to cycle-based.
    speedup  = proxy_metrics.get("real_speedup", sim_metrics["speedup"])
    area     = proxy_metrics["area_proxy"]
    power    = proxy_metrics["power_proxy"]
    t_viol   = 1.0 if proxy_metrics["timing_violation"] else 0.0
    overflow = proxy_metrics.get("acc_overflow", False)

    # Timing penalty (-3.0) is KEPT even though the effective-clock cap already
    # makes violations non-beneficial: requesting clk < critical_path gives the
    # same real_speedup as clk = critical_path (both clamp to the crit period)
    # AND a strictly higher power_proxy (power uses the requested, faster clock).
    # So the cap alone preserves the invariant "clk < crit never out-scores
    # clk = crit".  We retain the explicit -3.0 anyway because a timing
    # violation means the silicon literally will not function at the requested
    # clock — an infeasible design, not merely a no-benefit one — and the flag
    # plus penalty give the optimizer an unambiguous signal to leave that region.

    # Log2-scale speedup ∈ [−1, 1]: speedup=1.0 → 0.0,  real_speedup=514 → ≈0.98
    if speedup > 0 and max_spd > 1:
        norm_spd = math.log2(max(speedup, 1e-3)) / math.log2(max_spd)
        norm_spd = max(-1.0, min(1.0, norm_spd))
    else:
        norm_spd = -1.0

    # Hard floor: accelerator must beat SW by at least min_useful_speedup
    floor_penalty = perf_pen if speedup < min_spd else 0.0

    # Hard correctness penalties (sim-detected or proxy-predicted)
    correctness = 0.0
    if accuracy < 1.0:
        correctness -= 50.0 * (1.0 - accuracy)   # scale with error rate
    if overflow and accuracy >= 1.0:
        # Proxy says overflow but sim didn't catch it — shouldn't happen with
        # the updated sim, but guard against stale data.
        correctness -= 50.0

    return round(
        w_acc  * accuracy
        + w_spd  * norm_spd
        + w_area * area
        + w_pwr  * power
        + w_tv   * t_viol
        + correctness
        + floor_penalty,
        4,
    )

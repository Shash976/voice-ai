"""reward.py — proxy computations and multi-objective reward for the TinyMAC optimizer.

Reward formula
--------------
    reward = 2.0  * accuracy
           + 3.0  * log2_norm_speedup        # log2(speedup)/log2(200), ∈ [−1, 1]
           − 0.4  * area_proxy               # 1.0 at (8 lanes, 32b acc, 1024B bufs)
           − 0.4  * power_proxy              # 1.0 at baseline
           − 3.0  * timing_violation         # 0 or 1
           − 8.0  * perf_floor_penalty       # if speedup < min_useful_speedup (1.5×)
           − 50   * overflow_penalty         # if accumulator too narrow for TinyVAD
           − 50   * (1 − accuracy)           # if sim produces wrong outputs

Wall-clock sim time is NOT in the reward.  It is not a property of the design
and injecting it makes results non-reproducible across machines.

Design notes
------------
* accumulator_width is now a genuinely simulated parameter (sim_main.cpp clips
  the accumulator at runtime).  The proxy overflow check here is kept only as a
  fast pre-filter for agents that want to skip obvious misses without running
  the sim.

* area_proxy is properly normalised to 1.0 at the baseline config
  (mac_lanes=8, acc_width=32, input_buffer=1024B, weight_buffer=1024B).
  The MAC array is modelled as 80% of area, SRAM buffers as 20%.
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
_BASELINE_BUF_BYTES   = 1024 + 1024  # input + weight buffers


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
    """
    if area is None:
        area = area_proxy(config)
    clock_ns = config.get("clock_period_ns", 10)
    freq_mhz = 1000.0 / clock_ns
    return round(area * freq_mhz / 100.0, 4)  # 1.0 at baseline ✓


def timing_slack_ns(config: dict) -> float:
    """
    Timing slack = clock_period − estimated_critical_path  (ns).
    Positive → timing met; negative → violation (chip won't work at this clock).

    Critical path model (calibrated for ASAP7 7 nm):
      2.5 ns base adder + 0.15 ns routing per lane + 0.05 ns per acc register byte.

    At 16 lanes, 32b acc, 5 ns clock: slack = 5 − (2.5 + 2.4 + 0.2) = −0.1 ns  ← violation
    At  8 lanes, 32b acc, 5 ns clock: slack = 5 − (2.5 + 1.2 + 0.2) =  1.1 ns  ← just OK
    At  8 lanes, 32b acc, 10 ns clock: slack = 10 − 3.9 = 6.1 ns                ← comfortable
    """
    lanes    = config["mac_lanes"]
    acc_w    = config.get("accumulator_width", 32)
    clock_ns = config.get("clock_period_ns",   10)
    crit     = 2.5 + 0.15 * lanes + 0.05 * (acc_w / 8)
    return round(clock_ns - crit, 3)


def acc_overflows(config: dict) -> bool:
    """
    Fast proxy check: True if acc_width is analytically too narrow for TinyVAD.
    The sim will also catch this (accuracy < 1.0), but this lets agents skip
    obviously bad configs without launching a subprocess.
    """
    acc_w = config.get("accumulator_width", 32)
    return _TINYVAD_MAX_ACC > _INT_MAX.get(acc_w, _INT_MAX[32])


def compute_proxies(config: dict) -> dict:
    """Return all proxy metrics for a config dict (no sim required)."""
    a      = area_proxy(config)
    p      = power_proxy(config, a)
    slack  = timing_slack_ns(config)
    return {
        "area_proxy":       a,
        "power_proxy":      p,
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
    proxy_metrics : dict returned by compute_proxies() (area, power, slack, …)
    weights       : optional override dict (keys match search_space.yaml reward section)
    """
    w = weights or {}
    w_acc    = w.get("w_accuracy",          2.0)
    w_spd    = w.get("w_speedup",           3.0)
    w_area   = w.get("w_area",             -0.4)
    w_pwr    = w.get("w_power",            -0.4)
    w_tv     = w.get("w_timing_violation", -3.0)
    max_spd  = w.get("max_speedup",        200.0)
    min_spd  = w.get("min_useful_speedup",   1.5)
    perf_pen = w.get("perf_floor_penalty",  -8.0)

    accuracy = sim_metrics["accuracy"]
    speedup  = sim_metrics["speedup"]
    area     = proxy_metrics["area_proxy"]
    power    = proxy_metrics["power_proxy"]
    t_viol   = 1.0 if proxy_metrics["timing_violation"] else 0.0
    overflow = proxy_metrics.get("acc_overflow", False)

    # Log2-scale speedup ∈ [−1, 1]: speedup=1.0 → 0.0,  speedup=191 → ≈0.99
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

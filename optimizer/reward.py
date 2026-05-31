"""reward.py — multi-objective reward function for the TinyMAC optimizer.

All proxy computations live here so agents can evaluate area/power/timing
without running the full Verilator simulation.

Reward formula (higher is better):
    reward = 2.0 * accuracy
           + 1.5 * min(speedup / 200, 1.0)
           - 1.0 * area_proxy
           - 1.0 * power_proxy
           - 3.0 * timing_violation
           - 0.5 * (elapsed_s / 120)
           + correctness_penalty   (large negative if overflow or wrong outputs)
"""

from __future__ import annotations

# TinyVAD worst-case unsigned accumulator (Conv0: kernel=5, in_ch=40, out_ch=32)
# Each MAC: int8 * int8 → int16 product; K=200 products summed → int32 accumulator.
# Worst-case value: 200 * 127 * 127 = 3,226,600
_TINYVAD_MAX_ACC = 3_226_600

_INT_MAX = {16: 32_767, 24: 8_388_607, 32: 2_147_483_647}


# ── Proxy computations ────────────────────────────────────────────────────────

def area_proxy(config: dict) -> float:
    """
    Relative chip area, normalized so baseline config = 1.0.
    Baseline: mac_lanes=8, accumulator_width=32, buffers=1024B each.

    Area model:
      MAC array ∝ mac_lanes × accumulator_width
      Buffers   ∝ (input_buffer_bytes + weight_buffer_bytes)
    """
    lanes   = config["mac_lanes"]
    acc_w   = config.get("accumulator_width", 32)
    in_buf  = config.get("input_buffer_bytes", 1024)
    wt_buf  = config.get("weight_buffer_bytes", 1024)

    mac_area = (lanes * acc_w) / (8 * 32)                     # normalized MAC array
    buf_area = (in_buf + wt_buf) / 2048 * 0.25                # buffers ~25% of baseline
    return round(mac_area + buf_area, 4)


def power_proxy(config: dict, area: float | None = None) -> float:
    """
    Dynamic power proxy: switching activity × capacitance × frequency.
    Normalized to baseline (area=1.0, clock=10 ns) → power=1.0.
    """
    if area is None:
        area = area_proxy(config)
    clock_ns  = config.get("clock_period_ns", 10)
    freq_mhz  = 1000.0 / clock_ns
    return round(area * freq_mhz / 100.0, 4)


def timing_slack_ns(config: dict) -> float:
    """
    Estimated timing slack = clock_period − critical_path.
    Positive → timing met; negative → violation.

    Critical path model (calibrated for ASAP7):
      base (2.0 ns) + routing per lane (0.12 ns) + acc register depth (0.05 ns/byte)
    """
    lanes    = config["mac_lanes"]
    acc_w    = config.get("accumulator_width", 32)
    clock_ns = config.get("clock_period_ns", 10)

    crit = 2.0 + 0.12 * lanes + 0.05 * (acc_w / 8)
    return round(clock_ns - crit, 3)


def acc_overflows(config: dict) -> bool:
    """True if the chosen accumulator width cannot hold TinyVAD's worst-case value."""
    acc_w = config.get("accumulator_width", 32)
    return _TINYVAD_MAX_ACC > _INT_MAX.get(acc_w, _INT_MAX[32])


def compute_proxies(config: dict) -> dict:
    """Return all proxy metrics for a config dict (no sim required)."""
    a = area_proxy(config)
    p = power_proxy(config, a)
    slack = timing_slack_ns(config)
    overflow = acc_overflows(config)
    return {
        "area_proxy":       a,
        "power_proxy":      p,
        "timing_slack_ns":  slack,
        "timing_violation": slack < 0.0,
        "acc_overflow":     overflow,
    }


# ── Reward scalar ─────────────────────────────────────────────────────────────

def compute_reward(
    sim_metrics:   dict,
    proxy_metrics: dict,
    elapsed_s:     float,
    weights:       dict | None = None,
) -> float:
    """
    Compute multi-objective reward (higher is better).

    sim_metrics keys:   accuracy, speedup
    proxy_metrics keys: area_proxy, power_proxy, timing_violation, acc_overflow
    elapsed_s:          wall-clock seconds the sim took
    weights:            override reward weights (default = search_space.yaml values)
    """
    w = weights or {}
    w_acc   = w.get("w_accuracy",         2.0)
    w_spd   = w.get("w_speedup",          1.5)
    w_area  = w.get("w_area",            -1.0)
    w_pwr   = w.get("w_power",           -1.0)
    w_tv    = w.get("w_timing_violation", -3.0)
    w_cost  = w.get("w_sim_cost",        -0.5)
    max_spd = w.get("max_speedup",       200.0)
    max_sec = w.get("max_sim_time_s",    120.0)

    accuracy = sim_metrics["accuracy"]
    speedup  = sim_metrics["speedup"]
    area     = proxy_metrics["area_proxy"]
    power    = proxy_metrics["power_proxy"]
    t_viol   = 1.0 if proxy_metrics["timing_violation"] else 0.0
    overflow = proxy_metrics.get("acc_overflow", False)

    # Hard correctness penalty (dominates reward when active)
    correctness = 0.0
    if accuracy < 1.0:
        correctness -= 50.0 * (1.0 - accuracy)
    if overflow:
        correctness -= 50.0

    r = (
        w_acc  * accuracy
        + w_spd  * min(speedup / max_spd, 1.0)
        + w_area * area
        + w_pwr  * power
        + w_tv   * t_viol
        + w_cost * min(elapsed_s / max_sec, 1.0)
        + correctness
    )
    return round(r, 4)

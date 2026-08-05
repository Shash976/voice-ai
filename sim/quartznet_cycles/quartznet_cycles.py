#!/usr/bin/env python3
# quartznet_cycles.py — Stage C cycle + memory-bandwidth model for QuartzNet 15x5
#
# Run:
#   python3 sim/quartznet_cycles/quartznet_cycles.py                 # 10 s utterance
#   python3 sim/quartznet_cycles/quartznet_cycles.py --seconds 20
#   python3 sim/quartznet_cycles/quartznet_cycles.py --lanes 8 16 32 64 128
#
# ── What this is, and why it exists ──────────────────────────────────────────
#
# `accel_execute()` in sim/verilator/sim_main.cpp computes a whole matvec
# instantly against a magic 0-wait-state memory and fakes latency afterwards.
# That is fine for TinyVAD (14.6 KB of weights) and useless for QuartzNet, whose
# entire design problem IS the memory hierarchy: 18.85 MB of weights and ~48 MB
# of activation traffic per 10 s utterance.  docs/07_quartznet_pivot.md calls
# Stage C the designed off-ramp — no more RTL until the dataflow is priced.
#
# This tool needs neither Verilator nor a RISC-V toolchain.
#
# ── Where the schedule comes from ────────────────────────────────────────────
#
# NOT from a second copy of the tiling logic.  It shells out to
# firmware/quartznet/qn_schedule, which drives the real interpreter's tile
# walker (qn_walk -> qn_tile_hook) and dumps every tile as CSV.  The cycle model
# and the bit-exact C interpreter therefore cannot drift apart: they are the
# same loop nest.
#
# ── The model ────────────────────────────────────────────────────────────────
#
# Per tile, three costs are computed and the slowest wins (the design is assumed
# to overlap DMA with compute via double buffering, which is the only sane way
# to build it and the assumption this tool exists to test):
#
#  1. COMPUTE.  MAC ops use the formula rtl/tb verified against the RTL for
#     LANES in {1,2,4,8,16,32} x ACC_W in {24,32}:
#
#         cycles = n_outputs * (ceil(reduction / LANES) + 2) + 1
#
#     The +2 is ACCEL_CH_OVERHEAD (S_INIT_CH -> S_MAC -> S_REQ) and the +1 is the
#     Stage 7 drain cycle.  ADD/REQUANT are element-wise and carry no reduction,
#     so the formula does not apply; they are modelled as a LANES-wide
#     element-wise pass, ceil(n_outputs / LANES) + 3.  Flagged as a modelling
#     choice, not an RTL-verified number — they are 0.0% of the MACs.
#
#  2. ON-CHIP SRAM PORTS.  Activations and weights are staged in paired
#     fakeram45_1024x32 macros: docs/07_quartznet_pivot.md's banking decision is
#     "pair two 1024x32 side-by-side as one logical 64-bit port", giving 8 B per
#     bank pair per cycle.  A MAC consumes one activation byte and one weight
#     byte, so the array demands 2 * LANES B/cycle plus the output writes:
#
#         sram_cycles = ceil((2 * macs + out_bytes) / (8 * BANKS))
#
#     No operand-reuse credit is taken, so this is the pessimistic end.  It is
#     what makes LANES=64 interesting: 8 bank pairs sustain LANES=32 exactly.
#
#  3. EXTERNAL MEMORY, accumulated over the whole run rather than per tile,
#     because the DMA queue smooths it:
#       QSPI  flash: weights + qparams, read-only.
#       PSRAM      : activation reads and writes.
#
# Activation traffic is accounted straight off the tile stream and reproduces
# quartznet_topology.py::activation_traffic(fuse_dw_pw=True, retain_halo=True)
# — DW->PW fusion from the QN_F_FUSE_NEXT flag, halo retention from the tile's
# t_new span.  The sanity check at the bottom asserts that agreement against the
# figures committed in docs/07_quartznet_pivot.md.

from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FIRMWARE = ROOT / "firmware" / "quartznet"
SCHEDULE_BIN = FIRMWARE / "qn_schedule"
REF_DIR = ROOT / "build" / "quartznet"

OP_DW, OP_PW, OP_ADD, OP_REQUANT = 0, 1, 2, 3
OP_NAMES = {OP_DW: "dw", OP_PW: "pw", OP_ADD: "add", OP_REQUANT: "requant"}
F_FUSE_NEXT = 1 << 3

# ── Committed figures this tool must reproduce (docs/07_quartznet_pivot.md) ────
DOC_MACS_PER_FRAME = 18_847_040
DOC_WEIGHT_QPARAM_BYTES = 18_847_040 + 837_240        # 19.68 MB
DOC_ACT_BYTES_10S = 48_080_000                        # 48.08 MB, fuse + halo
DOC_TOTAL_MBPS_10S = 6.78
FPS_OUT = 50                                          # C1 stride 2 -> 50 fps
RT_TARGET_MMACS = 942.0                               # 18.847 M x 50 fps

# ── Defaults ──────────────────────────────────────────────────────────────────
DEF_FREQ_MHZ = 414.9      # measured pipelined-requantize Fmax, nangate45
DEF_BANKS = 8             # bank pairs of fakeram45_1024x32 -> 64 KB, 64 B/cycle
DEF_WT_BUF_KB = 64        # on-chip weight buffer (doc's "~64-96 KB" SRAM)
DEF_QSPI_MBPS = 52.0      # quad-SPI NOR flash @ 104 MHz = 4 bit x 104 M / 8
DEF_PSRAM_MBPS = 66.5     # APS6404L quad PSRAM @ 133 MHz


# ── Schedule acquisition ──────────────────────────────────────────────────────

def ensure_inputs(verbose: bool = True) -> pathlib.Path:
    """Make sure the descriptor table and the schedule dumper both exist."""
    desc = REF_DIR / "quartznet_desc.bin"
    if not desc.exists():
        if verbose:
            print(f"[build] emitting {desc.relative_to(ROOT)}", file=sys.stderr)
        subprocess.run([sys.executable, "quartznet_descriptors.py"],
                       cwd=ROOT / "sw" / "tinyml_reference", check=True,
                       stdout=subprocess.DEVNULL)
    if not SCHEDULE_BIN.exists():
        if verbose:
            print("[build] make -C firmware/quartznet schedule", file=sys.stderr)
        subprocess.run(["make", "schedule"], cwd=FIRMWARE, check=True,
                       stdout=subprocess.DEVNULL)
    return desc


def load_schedule(desc: pathlib.Path, t_out: int):
    """Drive the real interpreter's tile walker and parse what it emits."""
    out = subprocess.run([str(SCHEDULE_BIN), str(desc), str(t_out)],
                         check=True, capture_output=True, text=True).stdout

    model, descs, tiles = {}, [], []
    section = None
    header_pending = False
    for line in out.splitlines():
        if line.startswith("#"):
            section = line[1:]
            header_pending = True
            continue
        if header_pending:                 # skip each section's column header
            header_pending = False
            continue
        f = line.split(",")
        if section == "MODEL":
            model[f[0]] = f[1]
        elif section == "DESC":
            v = [int(x) for x in f]
            descs.append(dict(zip(
                ("idx", "op", "flags", "c_in", "c_out", "k", "stride", "dilation",
                 "pad", "in_buf", "out_buf", "res_buf", "in_stride", "out_stride",
                 "w_bytes", "q_bytes"), v)))
        elif section == "TILE":
            tiles.append(tuple(int(x) for x in f))
    return model, descs, tiles


# ── Traffic accounting, straight off the tile stream ──────────────────────────

def account_traffic(descs, tiles, t_out: int):
    """PSRAM activation bytes + per-descriptor MAC and time-tile counts.

    Reproduces activation_traffic(fuse_dw_pw=True, retain_halo=True):
      DW  reads only the frames not retained from the previous tile; its output
          write vanishes when QN_F_FUSE_NEXT says the next PW consumes it on chip.
      PW  skips its input read when it is the fused consumer of that DW.
      ADD reads both operands and writes the result; REQUANT reads and writes.
    """
    act_read = act_write = 0
    total_macs = 0
    macs_by_op = {OP_DW: 0, OP_PW: 0, OP_ADD: 0, OP_REQUANT: 0}
    time_tiles = [0] * len(descs)
    seen_t0 = [set() for _ in descs]

    for (di, op, t0, t1, c0, c1, red, n_out,
         _in0, _in1, new0, new1, _first, _last) in tiles:
        d = descs[di]
        if t0 not in seen_t0[di]:
            seen_t0[di].add(t0)
            time_tiles[di] += 1

        if op == OP_DW:
            act_read += (new1 - new0) * (c1 - c0)
            if not (d["flags"] & F_FUSE_NEXT):
                act_write += n_out
            macs = n_out * red
        elif op == OP_PW:
            prev = descs[di - 1] if di > 0 else None
            fused_in = (prev is not None and prev["op"] == OP_DW
                        and (prev["flags"] & F_FUSE_NEXT)
                        and prev["out_buf"] == d["in_buf"])
            if not fused_in:
                act_read += (t1 - t0) * d["c_in"]
            act_write += n_out
            macs = n_out * red
        elif op == OP_ADD:
            act_read += 2 * n_out
            act_write += n_out
            macs = 0
        else:                                    # OP_REQUANT
            act_read += n_out
            act_write += n_out
            macs = 0

        total_macs += macs
        macs_by_op[op] += macs

    return {
        "act_read": act_read,
        "act_write": act_write,
        "act_total": act_read + act_write,
        "macs": total_macs,
        "macs_by_op": macs_by_op,
        "time_tiles": time_tiles,
    }


def qspi_bytes(descs, time_tiles, wt_buf_bytes: int):
    """Weight + qparam bytes actually fetched from QSPI.

    A descriptor whose weights fit the on-chip weight buffer is fetched once and
    reused across its time tiles; otherwise it is re-streamed per time tile.
    """
    once = streamed = 0
    refetched = 0
    for d in descs:
        n = d["w_bytes"] + d["q_bytes"]
        once += n
        if n <= wt_buf_bytes:
            streamed += n
        else:
            streamed += n * time_tiles[d["idx"]]
            refetched += 1
    return once, streamed, refetched


# ── Cycle model ───────────────────────────────────────────────────────────────

def tile_cycles(tiles, lanes: int, banks: int):
    """Per-tile max(compute, on-chip SRAM port) cycles, summed."""
    port_bytes = 8 * banks                       # 64-bit port per bank pair
    total = compute = sram = 0
    for (_di, op, _t0, _t1, _c0, _c1, red, n_out,
         _i0, _i1, _n0, _n1, _f, _l) in tiles:
        if op in (OP_DW, OP_PW):
            c = n_out * (-(-red // lanes) + 2) + 1       # rtl/tb-verified
            sb = 2 * n_out * red + n_out                 # 1 act + 1 wt per MAC
        else:
            c = -(-n_out // lanes) + 3                   # element-wise pass
            sb = 3 * n_out
        s = -(-sb // port_bytes)
        compute += c
        sram += s
        # The two overlap inside a tile, so the tile costs whichever is slower.
        total += max(c, s)
    return total, compute, sram


# ── Report ────────────────────────────────────────────────────────────────────

def mb(x: float) -> str:
    return f"{x / 1e6:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage C cycle + memory-bandwidth model for QuartzNet 15x5")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="utterance length (default 10 s)")
    ap.add_argument("--t-out", type=int, default=None,
                    help="output frames (overrides --seconds)")
    ap.add_argument("--lanes", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--freq-mhz", type=float, default=DEF_FREQ_MHZ)
    ap.add_argument("--banks", type=int, default=DEF_BANKS,
                    help="paired fakeram45_1024x32 bank pairs (8 B/cycle each)")
    ap.add_argument("--wt-buf-kb", type=int, default=DEF_WT_BUF_KB)
    ap.add_argument("--qspi-mbps", type=float, default=DEF_QSPI_MBPS)
    ap.add_argument("--psram-mbps", type=float, default=DEF_PSRAM_MBPS)
    args = ap.parse_args()

    t_out = args.t_out if args.t_out else int(round(args.seconds * FPS_OUT))
    secs = t_out / FPS_OUT

    desc = ensure_inputs()
    model, descs, tiles = load_schedule(desc, t_out)
    tr = account_traffic(descs, tiles, t_out)
    once, streamed, refetched = qspi_bytes(descs, tr["time_tiles"],
                                           args.wt_buf_kb * 1024)

    print("=" * 78)
    print("QuartzNet 15x5 — Stage C cycle + memory model")
    print("=" * 78)
    print(f"  schedule    : {SCHEDULE_BIN.relative_to(ROOT)} "
          f"(drives qn_walk -> qn_tile_hook)")
    print(f"  descriptors : {model['n_desc']}   tiles {len(tiles):,}   "
          f"T_TILE {model['t_tile']}   DW_CH_TILE {model['dw_ch_tile']}")
    print(f"  utterance   : {secs:g} s = {t_out} output frames @ {FPS_OUT} fps")
    print(f"  clock       : {args.freq_mhz} MHz     "
          f"SRAM {args.banks} bank pairs = {args.banks * 8} B/cycle "
          f"({args.banks * 8} KB)")
    print(f"  weight buf  : {args.wt_buf_kb} KB      "
          f"QSPI {args.qspi_mbps} MB/s   PSRAM {args.psram_mbps} MB/s")

    # ── workload ──────────────────────────────────────────────────────────────
    macs_per_frame = tr["macs"] / t_out
    print("\n--- workload ---------------------------------------------------------")
    print(f"  MACs total          {tr['macs']:>18,}")
    print(f"  MACs per out frame  {macs_per_frame:>18,.0f}")
    for op in (OP_DW, OP_PW):
        share = 100.0 * tr["macs_by_op"][op] / tr["macs"]
        print(f"    {OP_NAMES[op]:<7}           {tr['macs_by_op'][op]:>18,}  "
              f"({share:.1f}%)")

    # ── traffic ───────────────────────────────────────────────────────────────
    print("\n--- external traffic -------------------------------------------------")
    print(f"  QSPI  weights+qparams, read once      {mb(once):>10} MB   (floor)")
    print(f"  QSPI  as scheduled @ {args.wt_buf_kb} KB wt buffer   "
          f"{mb(streamed):>10} MB   "
          f"({streamed / once:.1f}x; {refetched}/{len(descs)} descriptors re-streamed)")
    print(f"  PSRAM activation read                 {mb(tr['act_read']):>10} MB")
    print(f"  PSRAM activation write                {mb(tr['act_write']):>10} MB")
    print(f"  PSRAM total                           {mb(tr['act_total']):>10} MB")
    print(f"  ideal-floor aggregate                 "
          f"{mb(once + tr['act_total']):>10} MB   "
          f"= {(once + tr['act_total']) / 1e6 / secs:.2f} MB/s")
    print(f"  as-scheduled aggregate                "
          f"{mb(streamed + tr['act_total']):>10} MB   "
          f"= {(streamed + tr['act_total']) / 1e6 / secs:.2f} MB/s")

    # ── weight-buffer sensitivity ─────────────────────────────────────────────
    print("\n--- QSPI traffic vs on-chip weight-buffer capacity --------------------")
    print(f"  {'wt buf':>8} {'QSPI MB':>10} {'x floor':>8} {'re-streamed':>12} "
          f"{'QSPI s':>8}")
    fit_bytes = max(d["w_bytes"] + d["q_bytes"] for d in descs)
    fit_kb = math.ceil(fit_bytes / 1024)
    for kb in sorted({16, 32, 64, 128, 256, fit_kb, 512}):
        _, st, rf = qspi_bytes(descs, tr["time_tiles"], kb * 1024)
        print(f"  {kb:>6} KB {st / 1e6:>10.2f} {st / once:>8.2f} "
              f"{rf:>7}/{len(descs):<4} {st / 1e6 / args.qspi_mbps:>8.2f}")
    print("  (a descriptor whose weights fit the buffer is fetched once and reused")
    print("   across its time tiles; otherwise it is re-streamed per time tile)")
    print(f"  widest descriptor is {fit_bytes:,} B, so a {fit_kb} KB "
          f"weight buffer removes ALL re-streaming.")

    # ── LANES sweep ───────────────────────────────────────────────────────────
    freq = args.freq_mhz * 1e6
    psram_t = tr["act_total"] / (args.psram_mbps * 1e6)
    cyc_cache = {ln: tile_cycles(tiles, ln, args.banks) for ln in args.lanes}

    def sweep(title, qspi_b):
        qspi_t = qspi_b / (args.qspi_mbps * 1e6)
        print(f"\n--- LANES sweep — {title} "
              + "-" * max(0, 52 - len(title)))
        print(f"  {'LANES':>6} {'cycles':>15} {'compute s':>10} {'on-chip':>8} "
              f"{'wall s':>8} {'MMAC/s':>9} {'vs 942':>7} {'util':>7} {'RT':>5}"
              f"  bound")
        for lanes in args.lanes:
            cyc, ccyc, scyc = cyc_cache[lanes]
            comp_t = cyc / freq
            wall = max(comp_t, qspi_t, psram_t)
            bound = max((comp_t, "compute"), (qspi_t, "QSPI"),
                        (psram_t, "PSRAM"))[1]
            onchip = "MAC" if ccyc >= scyc else "SRAM"   # what limits the array
            mmacs = tr["macs"] / wall / 1e6
            util = 100.0 * tr["macs"] / (cyc * lanes)
            print(f"  {lanes:>6} {cyc:>15,} {comp_t:>10.3f} {onchip:>8} "
                  f"{wall:>8.3f} {mmacs:>9.1f} {mmacs / RT_TARGET_MMACS:>6.2f}x "
                  f"{util:>6.1f}% {secs / wall:>4.1f}x  {bound}")
        print(f"  external floors: QSPI {qspi_t:.3f} s, PSRAM {psram_t:.3f} s "
              f"(fully overlapped with compute)")

    sweep(f"{args.wt_buf_kb} KB weight buffer", streamed)
    if streamed > once:
        sweep(f"{fit_kb} KB weight buffer (QSPI at its floor)", once)

    print("\n  on-chip = whether the MAC formula or the SRAM ports set the tile cost;")
    print(f"            {args.banks} bank pairs deliver {args.banks * 8} B/cycle and "
          f"a MAC eats 2 B, so they sustain LANES <= {args.banks * 4}.")
    print("  util    = MACs / (cycles x LANES).  RT = utterance s / wall s; "
          ">= 1.0x is real-time.")

    # ── sanity checks against the committed figures ───────────────────────────
    print("\n--- sanity check vs docs/07_quartznet_pivot.md -----------------------")
    ok = True

    def chk(name, got, want, tol, unit=""):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name:<34} "
              f"got {got:,.2f}{unit}  expect {want:,.2f}{unit}")

    chk("MACs per output frame", macs_per_frame, DOC_MACS_PER_FRAME, 0.5)
    chk("weight + qparam bytes", once, DOC_WEIGHT_QPARAM_BYTES, 0)
    if abs(secs - 10.0) < 1e-9:
        chk("activation traffic, 10 s (MB)", tr["act_total"] / 1e6,
            DOC_ACT_BYTES_10S / 1e6, 0.02, " MB")
        chk("ideal-floor aggregate (MB/s)",
            (once + tr["act_total"]) / 1e6 / secs, DOC_TOTAL_MBPS_10S, 0.01,
            " MB/s")
    chk("real-time MAC target (MMAC/s)", macs_per_frame * FPS_OUT / 1e6,
        RT_TARGET_MMACS, 0.5, " MMAC/s")
    pw_share = 100.0 * tr["macs_by_op"][OP_PW] / tr["macs"]
    chk("pointwise share of MACs (%)", pw_share, 90.6, 0.1, "%")

    print(f"\n{'ALL SANITY CHECKS PASS' if ok else '*** SANITY CHECK FAILED ***'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

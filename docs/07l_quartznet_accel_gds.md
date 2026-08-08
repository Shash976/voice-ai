# 07l — First GDS for the QuartzNet accelerator (Stage E, begun)

The original Stage 7 design doc flagged physical implementation as Stage E,
"⚠️ highest risk," and `CLAUDE.md`'s "Open work" items 3-4 (the LANES resweep,
macro orientation) were deferred because this machine had no working
OpenROAD build. That changed — `~/OpenROAD-flow-scripts` is now a real,
working install (OpenROAD v26Q2, ORFS's own bundled Yosys 0.64). This session
produced the first-ever GDS for `rtl/accel/quartznet_accel.v`
(LANES=32, ACC_W=32 — the real design point).

## Result: clean pass, first attempt

```
Chip area:        120,653 µm²  (40% utilization)
DRC violations:   0
Timing (10.0 ns target): WNS 0.00 ns, TNS 0.00 ns — fully closed
Real critical path:      period_min = 6.62 ns → Fmax ≈ 150.96 MHz
Total power:       0.157 W (95% combinational — no macros, pure logic)
GDS:               59.0 MB (not committed — see below)
```

For scale: **8.3× the area** of the original TinyVAD accelerator's GDS
(14,518 µm² at LANES=4, `physical/orfs/measured/tinymac_accel_pristine/`) —
expected, since this design is LANES=32 with a full autonomous
descriptor-table sequencer, `OP_ADD` support, and an MMIO register file that
design never had.

**This is a fast bring-up configuration (`ABC_AREA=1`), not a final-quality
result** — see below. It exists to prove the mechanical pipeline
(synth→floorplan→place→CTS→route→GDS) works end to end for this design,
which had never been attempted, before committing to the much slower
high-quality path. It did, on the first attempt. A tighter/higher-quality
re-run is natural follow-on work, not required to call this "a GDS."

## The real blocker, found before any tool ran: `ext_mem_if.v` cannot be synthesized as-is

`rtl/accel/ext_mem_if.v`'s own header says so: **"NOT synthesizable, and
deliberately so."** Its `QSPI_BYTES`/`PSRAM_BYTES`-sized
`reg [7:0] mem[0:N-1]` arrays (25 MB + 1 MB at the real deployment
parameters) are a Verilator-only behavioral stand-in for what would become a
real QSPI controller + PSRAM PHY on actual silicon — that controller RTL was
never written, and neither was the on-chip SRAM macro wrapping
(`act_sram.v`/`wt_buf.v`) the original design doc planned but never built.
Feeding the real file to Yosys isn't slow, it's nonsensical — on the order
of 200 million flip-flops, and ORFS's own `synth.tcl` explicitly dumps a
`mem.json`/runs `mem_dump.py --max-bits` check specifically to "fail early
if this synthesis run is doomed."

**The chosen fix — approved after asking, but implemented differently than
initially approved, because the literal ask doesn't work in this ORFS**:
the user asked to "blackbox `ext_mem_if.v`, synthesize the rest." A literal
Yosys `(* blackbox *)` **does** elaborate and synthesize cleanly under ORFS's
Yosys 0.64 (measured) — but it doesn't survive the next step. ORFS's
`synth_odb.tcl`→`load.tcl` does `read_lef`/`read_verilog`/`link_design`
against the synthesized netlist, and a blackboxed cell has no LEF master to
link against. Writing one means hand-authoring an 805-pin LEF + `.lib` for a
macro whose Yosys-mangled instance name (parameterized blackboxes get
name-hashed, e.g. `\$paramod$f78...\ext_mem_if`) can't even be predicted in
advance.

**What was built instead: `physical/orfs/stubs/ext_mem_if_synth.v`** — a
tiny *synthesizable* module, still named `ext_mem_if` (so
`quartznet_accel.v`'s instantiation binds to it unmodified), with the real
module's exact port list and no real storage. Its only job is to keep every
port a real electrical load so surrounding logic isn't dead-code-eliminated,
at minimum area. One real correctness trap found building it: an early draft
only folded 8 of `q_req_addr`'s 32 bits into the response, and Yosys
correctly pruned the unused upper 24 bits — taking a large slice of
`addr_gen.v`'s real 32-bit address arithmetic down with it (79,698 cells
collapsed to 64,284). Fixed by XOR-folding all 32 bits of every port into the
stub's behavior, so nothing is prunable. **Verified the stub doesn't
over-prune or under-prune real logic two independent ways**: a true-blackbox
build (77,691 cells for the surrounding logic) and (stub build − stub
standalone) (77,774 cells) agree to **0.1%**.

This achieves the exact same *intent* the user approved — real accelerator
logic synthesized, the admittedly-non-synthesizable memory model excluded —
via a mechanism that actually works in this ORFS version. Full reasoning is
in the stub file's own header.

## Config

- `physical/orfs/make/nangate45/quartznet_accel/{config.mk,constraint.sdc}` —
  same 3-file-per-platform convention as `nangate45/tinymac_accel/`.
  `VERILOG_TOP_PARAMS = LANES 32 ACC_W 32` pins the design point explicitly
  rather than inheriting Verilog module defaults (belt-and-braces against a
  future default change silently moving the physical design point).
- **Clock target chosen from a real measurement, not a guess.** `tinymac_accel`'s
  269-415 MHz numbers don't transfer: Stage 7 already found the critical
  path moved off the (LANES-independent) Q31 requantize multiply onto the
  LANES-dependent MAC-accumulate path once requantize was pipelined, and
  this design runs LANES=32 (8× tinymac's 4) — nobody had measured LANES=32
  Fmax before this session. Started the SDC at a deliberately relaxed 10.0 ns,
  informed by a pre-layout OpenSTA pass on the mapped netlist (6.401 ns worst
  reg→reg, no wire RC/CTS yet) — leaving ~35% margin for routing so a first
  attempt tests the *flow*, not a timing guess. Post-route, the guess held
  with room to spare (WNS exactly 0.00, +3.38 ns worst slack) — the real
  period_min (6.62 ns) came out within 3% of the pre-layout estimate,
  suggesting a clean layout, not a lucky pass.
- `physical/orfs/make/run_quartznet.sh` — new driver script (not a
  generalized `run.sh`, matching the existing `run_spike.sh` precedent of
  one script per design). Stages the real RTL plus the synthesis stub
  (never the real `ext_mem_if.v` — actively deletes any stale copy before
  staging) into the gitignored `make/src/` work tree, then runs ORFS's
  classic make flow with `WORK_HOME` outside `make/`. `ORFS_DIR` defaults to
  `~/OpenROAD-flow-scripts` (this machine's actual path), not
  `/opt/OpenROAD-flow-scripts` (the company VM path `run.sh` assumes).

## What's not done yet

- **Full-quality synthesis** (drop `ABC_AREA=1`, let ORFS's default
  `abc_speed.script` run) — expected smaller area and comparable-or-better
  Fmax, at real time cost: the full script was independently measured at
  over 35 minutes and still running on this ~130k-gate netlist, vs. the
  bring-up script's few minutes.
- **Finding real Fmax** — the 10.0 ns target closed with +3.38 ns of slack
  to spare, meaning the true achievable period is close to the measured
  6.62 ns but not yet *targeted* directly. Next: re-run with `clk_period`
  walked down (8.0 → 7.0 → 6.5 ns per `constraint.sdc`'s own suggested
  ladder), reading `period_min = clk_period − wns` each time, matching the
  workflow `physical/orfs/measured/README.md` already documents for
  `tinymac_accel`.
- **The LANES resweep** (`CLAUDE.md` Open-work item 3) — this config plus
  `VERILOG_TOP_PARAMS` is exactly the vehicle for it, now that a working
  baseline exists.
- **asap7** — this run is nangate45 only, matching the original design's own
  precedent of validating on the easier PDK first.
- **The GDS binary itself is not committed** (59 MB; `physical/orfs/runs/`
  is gitignored, matching the existing convention — only curated text
  reports are checked in, at `physical/orfs/measured/quartznet_accel_bringup/`).

## Verification

```
$ physical/orfs/make/run_quartznet.sh nangate45
...
Chip area for module '\quartznet_accel': 120653.078000
Found and reported 0 problems.
$ wc -c physical/orfs/runs/reports/nangate45/quartznet_accel/base/5_route_drc.rpt
0 ...
$ grep -E "^wns|^tns|period_min" physical/orfs/runs/reports/nangate45/quartznet_accel/base/6_finish.rpt
wns max 0.00
tns max 0.00
core_clock period_min = 6.62 fmax = 150.96
```

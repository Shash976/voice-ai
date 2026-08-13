# physical/orfs — Stage 6/7: RTL-to-GDS

Pushes synthesizable accelerator cores ([`rtl/accel/`](../../rtl/accel)) through
OpenROAD-flow-scripts (Yosys synthesis → OpenROAD floorplan / place / CTS /
route → GDS) to get real area, timing, and power numbers. Two designs live
here: the original Stage 6 TinyVAD core (`tinymac_accel`) and the Stage 7
QuartzNet ASR core (`quartznet_accel`).

## What gets synthesized

**`tinymac_accel`** — the int8 MAC array, accumulator, and requantize
datapath (plus its sequencer FSM). Not the PicoRV32, not main RAM. This is
the block whose area/timing the Stage-5 knobs (`LANES`, `ACC_W`) actually
move, and it is small and synchronous — the safe choice for a first GDS
(project plan, Stage 6 "Option A"). Default parameters are the Stage-5 grid
optimum: `LANES=4`, `ACC_W=24`.

**`quartznet_accel`** — the Stage 7 descriptor-table-driven accelerator
(`addr_gen`/`int8_mac_array`/`requantize`/`requantize_add`, autonomous
table-walk sequencer), LANES=32 ACC_W=32 (the real design point). Its
`ext_mem_if.v` memory model is deliberately non-synthesizable (a Verilator
behavioral stand-in for a QSPI/PSRAM controller never built as RTL), so this
flow substitutes a real-logic-preserving synthesis stub in its place — see
`physical/orfs/stubs/ext_mem_if_synth.v` and
[`docs/07l_quartznet_accel_gds.md`](../../docs/07l_quartznet_accel_gds.md)
for the full reasoning and first measured results (120,653 µm², 0 DRC,
150.96 MHz bring-up config).

## Synthesis (works offline — start here)

Real gate count + cell area with just Yosys + the on-disk PDK liberty. No
network, no OpenROAD:

```bash
physical/orfs/synth_area.sh sky130hd     # → reports/sky130hd_{area.rpt,synth.log,netlist.v}
physical/orfs/synth_area.sh nangate45
```

Latest numbers (LANES=4, ACC_W=24): sky130hd = 10,179 cells / 72,897 µm²;
nangate45 = 12,032 cells / 14,518 µm². See [`docs/06_rtl_to_gds.md`](../../docs/06_rtl_to_gds.md).

## Place & route → GDS (`make/` — the working flow)

The full flow runs through the **classic ORFS make flow** against a real ORFS
install (the company VM has one at `/opt/OpenROAD-flow-scripts`). A design is
just three files per platform — `config.mk`, `constraint.sdc`, and the RTL —
under `make/<platform>/tinymac_accel/`.

```bash
physical/orfs/make/run.sh                       # tinymac_accel, nangate45, through GDS
physical/orfs/make/run.sh nangate45 gui_final   # + open the OpenROAD GUI
physical/orfs/make/sweep.sh                     # LANES sweep → sweep_results.csv

physical/orfs/make/run_quartznet.sh              # quartznet_accel, nangate45, through GDS
physical/orfs/make/run_quartznet.sh nangate45 synth   # stop after synthesis
```

Platforms configured: **nangate45** (45 nm, primary), **asap7** (7 nm-class
target — note its SDC time unit is *picoseconds*; the optimizer handles the
conversion, see `optimizer/physical_runner.py`), **sky130hd** (130 nm bring-up).

The Stage-5 optimizer drives this flow programmatically — it now lives in the
standalone [eda-rl](https://github.com/Shash976/eda-rl) repo (a design-agnostic
multi-fidelity funnel optimizer that calls this same ORFS make flow).

> **Historical note:** a bazel-orfs route was tried first and abandoned — its
> gallery workspace needs PyPI access that times out on the available networks.
> Its files (`BUILD.bazel`, `sync.sh`, a root-level `constraints.sdc`) were
> removed; the make flow is the only supported path.

## Files

| File | Purpose |
|------|---------|
| `make/<platform>/tinymac_accel/config.mk` | ORFS design config (RTL list, clock, util/density) |
| `make/<platform>/tinymac_accel/constraint.sdc` | clock definition (platform-native time unit) |
| `make/run.sh` | stage tinymac_accel files into ORFS and run one full flow |
| `make/sweep.sh` | parameter sweep via `VERILOG_TOP_PARAMS` + per-config `FLOW_VARIANT` |
| `make/<platform>/quartznet_accel/config.mk` | ORFS design config for the QuartzNet accelerator (LANES=32 ACC_W=32) |
| `make/<platform>/quartznet_accel/constraint.sdc` | clock definition for `quartznet_accel` |
| `make/run_quartznet.sh` | stage quartznet_accel files (+ the `ext_mem_if` synthesis stub) and run one full flow |
| `stubs/ext_mem_if_synth.v` | synthesizable stand-in for the deliberately non-synthesizable `rtl/accel/ext_mem_if.v` |
| `synth_area.sh` | yosys-only area sweep, runs anywhere |

## Results

See [`docs/06_rtl_to_gds.md`](../../docs/06_rtl_to_gds.md) for collected metrics
and the comparison against Stage-5 cycle/area estimates.

# physical/orfs — Stage 6: RTL-to-GDS

Pushes the synthesizable TinyVAD accelerator core ([`rtl/accel/`](../../rtl/accel))
through OpenROAD-flow-scripts (Yosys synthesis → OpenROAD floorplan / place / CTS
/ route → GDS) to get real area, timing, and power numbers.

## What gets synthesized

Just the **accelerator compute core** `tinymac_accel` — the int8 MAC array,
accumulator, and requantize datapath (plus its sequencer FSM). Not the PicoRV32,
not main RAM. This is the block whose area/timing the Stage-5 knobs (`LANES`,
`ACC_W`) actually move, and it is small and synchronous — the safe choice for a
first GDS (project plan, Stage 6 "Option A").

Default parameters are the Stage-5 grid optimum: `LANES=4`, `ACC_W=24`.

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
physical/orfs/make/run.sh                       # one config, nangate45, through GDS
physical/orfs/make/run.sh nangate45 gui_final   # + open the OpenROAD GUI
physical/orfs/make/sweep.sh                     # LANES sweep → sweep_results.csv
```

Platforms configured: **nangate45** (45 nm, primary), **asap7** (7 nm-class
target — note its SDC time unit is *picoseconds*; the optimizer handles the
conversion, see `optimizer/physical_runner.py`), **sky130hd** (130 nm bring-up).

The Stage-5 optimizer drives this flow programmatically
(`optimizer/run_physical_optimizer.py`, and the multi-fidelity funnel in
`optimizer/funnel.py` — see [`docs/08_funnel_optimizer.md`](../../docs/08_funnel_optimizer.md)).

> **Historical note:** a bazel-orfs route was tried first and abandoned — its
> gallery workspace needs PyPI access that times out on the available networks.
> Its files (`BUILD.bazel`, `sync.sh`, a root-level `constraints.sdc`) were
> removed; the make flow is the only supported path.

## Files

| File | Purpose |
|------|---------|
| `make/<platform>/tinymac_accel/config.mk` | ORFS design config (RTL list, clock, util/density) |
| `make/<platform>/tinymac_accel/constraint.sdc` | clock definition (platform-native time unit) |
| `make/run.sh` | stage design files into ORFS and run one full flow |
| `make/sweep.sh` | parameter sweep via `VERILOG_TOP_PARAMS` + per-config `FLOW_VARIANT` |
| `synth_area.sh` | yosys-only area sweep, runs anywhere |

## Results

See [`docs/06_rtl_to_gds.md`](../../docs/06_rtl_to_gds.md) for collected metrics
and the comparison against Stage-5 cycle/area estimates.

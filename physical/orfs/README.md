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

Real gate count + cell area with just Yosys + the on-disk PDK liberty. No bazel,
no network, no OpenROAD:

```bash
physical/orfs/synth_area.sh sky130hd     # → reports/sky130hd_{area.rpt,synth.log,netlist.v}
physical/orfs/synth_area.sh nangate45
```

Latest numbers (LANES=4, ACC_W=24): sky130hd = 10,179 cells / 72,897 µm²;
nangate45 = 12,032 cells / 14,518 µm². See [`docs/06_rtl_to_gds.md`](../../docs/06_rtl_to_gds.md).

## Place & route → GDS (needs OpenROAD)

> ⚠️ Blocked on this machine: the bazel-orfs gallery pulls Python deps from PyPI
> (network times out here) and there is no standalone OpenROAD binary installed.
> The design files below are ready; run them once OpenROAD/network is available.

OpenROAD is provided by **bazel-orfs**, which is already cloned and cached at
`~/bazel-orfs`. The flow runs inside that repo's `gallery/` workspace (a working
bzlmod consuming project with the OpenROAD toolchain + ORFS PDKs wired up). The
files here are the canonical copies; `sync.sh` drops them into a `tinymac`
package there.

```bash
# 1. copy canonical design files into the gallery workspace
physical/orfs/sync.sh

# 2. run the flow (from the gallery workspace)
cd ~/bazel-orfs/gallery
bazel build //tinymac:tinymac_accel_synth    # synthesis: gate count, cell area
bazel build //tinymac:tinymac_accel_route     # through routing: WNS/TNS, wirelength
bazel build //tinymac:tinymac_accel_final      # full flow: GDS

# stage outputs land in ~/bazel-orfs/gallery/bazel-bin/tinymac/results/<pdk>/...
# logs + metrics + reports under the stage's _deps dir / results tree.
```

## PDK targets

- **sky130hd** (current `BUILD.bazel`) — 130 nm bring-up. Proven on this machine
  (the `gallery/serv` example runs here). Confirms the RTL synthesizes and the
  flow closes; gives a first area number.
- **asap7** — the project's real 7nm-class target. Switch `pdk` to
  `@orfs//flow:asap7` and tighten `constraints.sdc` to the 5 ns Stage-5 target,
  then report achieved slack (WNS/TNS) and area.

## Files

| File | Purpose |
|------|---------|
| `BUILD.bazel` | `demo_flow()` target — PDK, SDC, RTL list, util/density args |
| `constraints.sdc` | clock definition + I/O delays |
| `sync.sh` | copy canonical RTL + these files into the gallery `tinymac` package |
| `reports/` | collected area/timing/power reports (committed per run) |

## Results

See [`docs/06_rtl_to_gds.md`](../../docs/06_rtl_to_gds.md) for collected metrics
and the comparison against Stage-5 cycle/area estimates.

# physical/orfs/make — classic ORFS flow (company VM)

Drives the full RTL-to-GDS flow for `tinymac_accel` using the system OpenROAD at
`/opt/OpenROAD-flow-scripts` (the classic `make` flow IT set up — no bazel, no
PyPI). This is the path that produces real **WNS/TNS, power, and a GDS layout**.

## Layout (mirrors ORFS `designs/<platform>/<design>`)

```
physical/orfs/make/                    ← run from here (the working dir)
├── run.sh                             ← stages RTL + invokes the ORFS Makefile
├── nangate45/tinymac_accel/           ← config.mk + constraint.sdc (2.0 ns)
├── sky130hd/tinymac_accel/            ← config.mk + constraint.sdc (10 ns)
└── src/tinymac_accel/                 ← RTL, auto-staged from ../../../rtl/accel
```

`run.sh` copies the canonical RTL from `rtl/accel/` into `src/tinymac_accel/`
each run, so the Verilog never diverges from the verified source.

## Run it (on the VM)

```bash
cd physical/orfs/make
./run.sh                       # full flow on nangate45  → GDS
./run.sh nangate45 gui_final   # full flow, then open the OpenROAD GUI
./run.sh sky130hd              # full flow on sky130hd
./run.sh nangate45 synth       # stop after synthesis (quick area check)
```

Equivalent to the manual commands IT documented:
```bash
source /opt/OpenROAD-flow-scripts/env.sh
make --file=/opt/OpenROAD-flow-scripts/flow/Makefile \
     DESIGN_CONFIG=./nangate45/tinymac_accel/config.mk
```

## Where the results land

Under the working dir (`physical/orfs/make/`):

| Path | What |
|------|------|
| `logs/<plat>/tinymac_accel/base/*.log` | per-stage logs (synth, place, route…) |
| `reports/<plat>/tinymac_accel/base/` | **area, WNS/TNS, power** (`6_report.*`) |
| `results/<plat>/tinymac_accel/base/6_final.gds` | the **GDS** — open in `klayout` |

Quick metric peek after a run:
```bash
grep -E "wns|tns|Design area|Total Power" reports/nangate45/tinymac_accel/base/6_report.* 2>/dev/null
```

## Sweeping configurations (`sweep.sh`)

To see *many* chip configs compared — not just one — run the sweep. It drives the
same flow once per config, overriding the RTL parameters (`LANES`, `ACC_W`) via
ORFS `VERILOG_TOP_PARAMS` and the clock via a generated SDC, giving each config
its own `FLOW_VARIANT` (separate results + GDS):

```bash
physical/orfs/make/sweep.sh                  # default grid on nangate45
physical/orfs/make/sweep.sh sky130hd          # different PDK
CONFIGS_FILE=mygrid.txt physical/orfs/make/sweep.sh   # custom "LANES ACC_W CLK_NS" rows
```

Output: a `sweep_results.csv` + printed table of area / util / WNS / power / Fmax
per config, and one openable layout each:
```
klayout results/nangate45/tinymac_accel/<variant>/6_final.gds
```

This is **grid search** — the simplest design-space optimizer. The Stage-5 agents
(`optimizer/`) are the smarter version: they choose *which* configs to try using a
reward function instead of brute-forcing the whole grid.

## Targeting asap7 (the project's real PDK)

Copy `nangate45/tinymac_accel/` to `asap7/tinymac_accel/`, set `PLATFORM = asap7`
in its `config.mk`, and set a 7nm-appropriate `clk_period` in its
`constraint.sdc` (ASAP7 cell delays are small — start near the Stage-5 target and
read back WNS). Then `./run.sh asap7`.

## Notes / gotchas

- **`source env.sh` first** — `run.sh` does it for you; if running `make`
  manually, source it once per terminal so `yosys`/`openroad`/`klayout` are found.
- **Timing failures don't stop the flow** — ORFS finishes and reports negative
  WNS. To find Fmax, lower `clk_period` until WNS just crosses zero.
- **Clock port is `clk`, reset is `rst_n`** (active-low) — matches the SDC.
- The whole repo (or at least `rtl/accel/` + this dir) must be present on the VM
  so `run.sh` can find the RTL.

See [`../../../docs/06_rtl_to_gds.md`](../../../docs/06_rtl_to_gds.md) for the
design background and the synthesis-only numbers already collected.

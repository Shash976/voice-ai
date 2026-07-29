# Committed ORFS measurement artifacts

`physical/orfs/runs/` is gitignored (it reached 885 MB). These are the curated
text reports that back the numbers quoted in `docs/07_quartznet_pivot.md` and
`docs/07a_sram_macro_notes.md`, so the claims stay checkable after the run
directory is gone or the work moves to another machine.

All measured on nangate45 with the ORFS bundled Yosys 0.64 / OpenROAD, 2026-07-28.

| directory | design | what it shows |
|---|---|---|
| `tinymac_accel_pristine/` | LANES=4 ACC_W=24, combinational requantize | WNS −1.72 ns → period_min 3.72 ns → **268.8 MHz**; 16,723 µm² |
| `tinymac_accel_pipelined/` | same, 2-stage pipelined requantize | WNS −0.41 ns → period_min 2.41 ns → **414.9 MHz**; 17,271 µm² |
| `sram_spike/` | 4 fakeram45 macros (16 KB) | **0 DRC**, WNS 0.00 ns, 68,770 µm² @ 40% util |

Both `tinymac_accel` variants were run **on the same machine with the same
2.0 ns SDC**, so the 1.54× Fmax comparison is apples-to-apples. (The pristine
WNS also reproduces the −1.72 ns previously recorded on a different machine,
which is independent evidence that the flow is reproducible.)

## Files per directory

| file | contents |
|---|---|
| `6_finish.rpt` | final timing (`wns max` / `tns max`), worst paths, power |
| `synth_stat.txt` | per-cell-type area breakdown and total chip area |
| `synth_check.txt` | yosys structural checks |
| `5_route_drc.rpt` | detailed-route DRC violations (0 bytes = clean) |
| `2_floorplan_final.rpt` | post-floorplan area/utilization, macro placement |

## Reading the timing numbers

The SDC targets 2.0 ns, so `period_min = 2.0 − wns` and `Fmax = 1000 / period_min`:

```bash
grep -E "^wns|^tns" tinymac_accel_pipelined/6_finish.rpt
```

To regenerate any of these, see `docs/07b_machine_setup.md`.

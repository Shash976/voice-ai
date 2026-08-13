# 07j — The combined milestone: real weights, on real RTL, real English

This is the stretch goal from the original planning doc's "Combined" gate
(`~/.claude/plans/generate-an-implementation-plan-pure-robin.md`): *"full-config
RTL demo on a ~1s clip → legible transcript out of `sim/verilator_qn
MODE=rtl`"* — with Step 6's **real, NeMo-trained, calibrated int8 weights**
substituted in, not the seeded-random placeholder Step 1 (Gap 1's RTL demo
close-out) validated against.

## Result

```
$ make -C sim/verilator_qn run MODE=rtl CONFIG=full \
      GOLDEN_DIR=$(pwd)/build/quartznet_real T_OUT=70
...
[sim] Reset released -- starting simulation (RTL mode)
cycles=742545
transcript("as for etching")
[sim] Done in 554540208 cycles
```

**Real English, out of the real Verilog RTL** (`rtl/soc/qn_soc.v` +
`rtl/accel/quartznet_accel.v`, Verilated and simulated cycle-by-cycle — no
C++ behavioral emulation), on real calibrated weights, matching the C
interpreter's own transcript on the same input exactly (`docs/07g`'s
`"as for etching"`, from the first 1.4s of
`librispeech/LibriSpeech/dev-clean/1272/128104/1272-128104-0008.flac`).

Cross-checks, mirroring Step 1's three-way gate:
1. No `TIMEOUT`/`CPU TRAP` — clean completion.
2. RTL transcript == the C interpreter / NumPy golden transcript (both
   `"as for etching"`).
3. Cycle count (554,540,208) is within 0.0005% of Step 1's seeded-random run
   on the same config (554,542,940 cycles) — expected, since the FSM's
   cycle count is structural (descriptor shapes, tiling, memory traffic
   pattern), not data-dependent; the tiny residual difference is plausibly
   from weight-value-dependent effects like dead-channel branches, not
   investigated further since it's three orders of magnitude below anything
   that would matter.

## Zero code changes needed

This required no source changes at all — `git status` after the run is
clean. Step 1's fix (forwarding `QSPI_BYTES`/`PSRAM_BYTES` to the RTL
Verilate flags) and Step 5's exporter (`quartznet_export_int8.py`, which
already emits `quartznet_desc.bin`/`quartznet_weights.bin`/
`quartznet_qparams.bin`/`quartznet_input.bin` in exactly the layout
`sim/verilator_qn`'s `GOLDEN_DIR` mechanism expects) already composed
correctly the first time they were pointed at each other:

```bash
make -C sim/verilator_qn run MODE=rtl CONFIG=full \
    GOLDEN_DIR=$(pwd)/build/quartznet_real T_OUT=70
```

`GOLDEN_DIR` overrides the default seeded-random `build/quartznet`;
everything downstream (`gen_fw_headers.py`, the RV32 firmware build, the
QSPI backdoor preload) only ever reads `quartznet_desc.bin`/
`quartznet_input.bin` from whatever directory it's pointed at, with no
assumption baked in about seeded-random vs. real weights.

## What this closes out

This is the last item from the original plan's Gap 1 / Gap 2 split still
open. Both gaps' respective final claims are now demonstrated on the same
artifact:
- Gap 1's claim ("real Verilog RTL, not a C++ behavioral stand-in, drives
  the accelerator over a real bus") — proven since Stage D increment 5.
- Gap 2's claim ("the chip transcribes real speech using real trained
  weights") — proven since A5/A6, on the C interpreter and the firmware
  binary.
- **This result is the conjunction of both**: real RTL *and* real weights,
  together, for the first time.

## Verification

```
$ make -C firmware/quartznet host
...
PASS — 2/2 configurations bit-exact   # unaffected regression

$ make -C firmware/quartznet host-real
...
PASS — 1/1 configurations bit-exact   # unaffected regression
```

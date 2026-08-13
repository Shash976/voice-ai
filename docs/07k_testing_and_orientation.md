# 07k — Testing guide & repo orientation (Stage 7, QuartzNet)

A practical companion to the Stage 7 build docs (`07_quartznet_pivot.md`
through `07j_combined_milestone_real_weights_on_rtl.md`): what each folder
actually contains, and the exact commands to run your own audio through the
pipeline — from a fast software check up to the real chip RTL simulation.

Written for the QuartzNet 15x5 ASR accelerator specifically. For the
earlier TinyVAD wake-word chip (Stages 1-6, a different, older design),
see `00_onboarding_overview.md` and `01`-`06`.

---

## Where is the actual chip design?

**`rtl/`** — this is it, the real Verilog:

```
rtl/
  picorv32/picorv32.v          the CPU (open-source RISC-V core, unmodified)
  soc/
    picorv32_soc.v             CPU + RAM + UART, the base SoC (Stage 3-4)
    qn_soc.v                   wraps picorv32_soc.v + quartznet_accel.v with
                                real bus decode -- THE CHIP, as a whole
  accel/
    quartznet_accel.v          the accelerator itself: descriptor-driven
                                sequencer, MMIO register file, dispatches
                                OP_DW/OP_PW/OP_ADD/OP_REQUANT
    int8_mac_array.v           the MAC datapath (LANES x int8 multiply-add)
    requantize.v                per-channel int8 requantization (pipelined)
    requantize_add.v            the ADD op's 3-multiplier requantization
    addr_gen.v                   address generation for DW/PW tiling
    ext_mem_if.v                 QSPI (weights) + PSRAM (activations) interface
  tb/                          Verilator unit testbench (rtl/tb/quartznet_tb.cpp)
```

`qn_soc.v` is the top-level design — real bus decode, no C++ behavioral
stand-in. It's what `sim/verilator_qn`'s `MODE=rtl` Verilates and simulates.

## Where is the GDS? (Short answer: there isn't one for this chip yet)

**No GDS exists anywhere in this repo for `quartznet_accel.v` / `qn_soc.v`.**
Physical implementation (synthesis → place & route → GDS) has only ever been
run on two *other* things:

1. **The pre-QuartzNet accelerator** (`rtl/accel/tinymac_accel.v` — Stage 4-6's
   original TinyVAD-era MAC accelerator, a different, much smaller design).
   This one has a real completed flow: nangate45 GDS (~19,738 µm², Fmax
   ≈269 MHz) and a first asap7 data point (1,433 µm², Fmax 509 MHz). Text
   reports (not the GDS binary itself, which is never committed) live at
   `physical/orfs/measured/tinymac_accel_{pristine,pipelined}/`.
2. **A generic SRAM macro integration feasibility spike**
   (`rtl/spike_sram/sram_spike.v`) — proved fakeram45 SRAM macros can be
   placed and routed cleanly (0 DRC) on *some* design, as a derisking step
   before committing to the real QuartzNet accelerator's SRAM-heavy
   floorplan. Reports at `physical/orfs/measured/sram_spike/`. This is not
   `quartznet_accel.v` either.

`grep -rl quartznet_accel physical/orfs/make` returns nothing — the real
accelerator has never been pointed at the physical flow at all. This is
tracked, explicit, open work (`CLAUDE.md`'s Stage 7 "Open work" items 3-4:
the LANES resweep and macro-orientation experiment), blocked because this
machine has no working OpenROAD binary (`~/OpenROAD-flow-scripts` is
checked out but its build stopped at 81%). Everything this whole session
built and verified — the descriptor format, the C interpreter, the ONNX
calibration pipeline, and the RTL itself — has been checked via **Verilator
simulation** (cycle-accurate software modeling of the RTL's logic), never
through real silicon-target synthesis.

`physical/` in general holds the ORFS (OpenROAD-flow-scripts) flow
configs (`physical/orfs/make/{nangate45,asap7,sky130hd}/`) that *would*
drive that flow once run on a machine with OpenROAD built.

---

## Folder map

```
rtl/            the actual Verilog -- see above
firmware/
  picorv32_baremetal/   bare-metal startup/linker, shared by both firmware images
  quartznet/            the QuartzNet firmware + host-side C tooling:
                         quartznet_infer.c   the bit-exact int8 interpreter
                                              (same algorithm as the RTL,
                                              compiled for the RV32 target
                                              AND natively for x86 testing)
                         qn_main.c/qn_accel.c   RV32 hardware-dispatch driver
                         qn_transcribe.c      host-native mp3-to-text tool
                         test_quartznet_host.c  bit-exactness gate (C vs
                                              NumPy golden)
                         Makefile              all the `make` targets below
  tinyengine_port/      TinyVAD-era (Stage 1-4), unrelated to QuartzNet
sim/
  verilator/            Stage 3-4 TinyVAD sim (PicoRV32 + old accelerator)
  verilator_qn/         THE QuartzNet chip simulator -- MODE=shim (fast, C++
                         emulates the accelerator) or MODE=rtl (real
                         qn_soc.v, Verilated and simulated cycle-by-cycle)
  quartznet_cycles/     analytical cost model (cycles/bandwidth), not a sim
sw/tinyml_reference/    all the Python: NeMo checkpoint export, the audio
                         front end, the NumPy golden interpreter, ONNX
                         calibration/quantization, WER scoring
build/                  ALL generated artifacts -- gitignored, never
                         committed, always reproducible from source. This is
                         where your own test runs will land
                         (build/quartznet_real/ = the real calibrated model;
                         build/quartznet_int8/, build/quartznet_nemo/ =
                         calibration/export intermediates)
docs/07*.md             the Stage 7 build log, one doc per session/gate --
                         07_quartznet_pivot.md is the original design doc,
                         07d onward are this and later sessions' work
librispeech/             downloaded LibriSpeech dev-clean/test-clean, used
                         for calibration and WER evaluation (gitignored)
physical/                ORFS physical-design flow configs (see above)
```

---

## Getting started: run your own audio through it

Three ways, from fastest/least-real to slowest/most-real. All three run the
exact same real, NeMo-trained, calibrated int8 weights
(`build/quartznet_real/`) -- they differ only in *what executes the math*.

### 1. Fastest: native C interpreter (seconds)

Compiles `quartznet_infer.c` (the same algorithm as the RTL) to run
natively on your CPU. No hardware simulation at all -- use this to quickly
check a clip sounds sane before committing to a slow RTL run.

```bash
cd ~/voice-ai
make -C firmware/quartznet transcribe-real MP3=/path/to/clip.mp3
```

Clips over ~47s will hit the host build's fixed activation-arena size
(`QN_MAX_T_OUT`); truncate with a manual two-step call if needed:
```bash
python3 sw/tinyml_reference/mp3_to_text.py clip.mp3 --seconds 20 \
    --model-dir build/quartznet_real --out-dir build/quartznet_real
./firmware/quartznet/qn_transcribe build/quartznet_real build/quartznet_real/mp3_input.bin
```

### 2. Reference check: the NumPy golden model (seconds)

Same descriptor-table algorithm again, in plain Python/NumPy this time --
this is what the C interpreter and the RTL are both proven bit-exact
against. Useful as a quick preview / sanity cross-check.

```bash
python3 sw/tinyml_reference/quartznet_ref.py --weights-from build/quartznet_real
```

### 3. The real thing: actual chip RTL, simulated (minutes)

This Verilates `rtl/soc/qn_soc.v` (the real accelerator, no C++ stand-in)
and runs it cycle-by-cycle. This is Verilator -- a *software simulation* of
the hardware, not real silicon -- so it's slow (minutes, not the chip's
real sub-second speed), but it's the actual RTL executing your audio.

```bash
# Step 1: quantize your mp3 with the real calibrated weights (needs the venv)
venv/bin/python3 sw/tinyml_reference/quartznet_export_int8.py \
    --audio /path/to/clip.mp3 --t-out N --out build/quartznet_mytest
#   N ~= seconds_of_audio * 50, rounded up. Two independent limits bound N --
#   see the "two limits" note below before picking a long clip.

# Step 2: build the NumPy golden (writes quartznet_golden.bin, which the
# next step's Makefile rule requires as a trigger file -- it doesn't read
# its contents, just needs it to exist)
python3 sw/tinyml_reference/quartznet_ref.py --weights-from build/quartznet_mytest

# Step 3: Verilate + simulate the real RTL
make -C sim/verilator_qn run MODE=rtl CONFIG=full \
    GOLDEN_DIR=$(pwd)/build/quartznet_mytest T_OUT=N
```

Watch for the log's `transcript("...")` line. It should match step 2's
transcript byte-for-byte -- that agreement is the actual proof the chip
design is correct, not just the software model.

**Two independent limits bound how long a clip you can run, and they must
both be checked** (getting only the first one right still silently fails,
found the hard way running an 8s clip through `T_OUT=400`):

1. **PSRAM size** (`PSRAM_BYTES`, default 1MB) bounds the activation arena
   -- default supports `T_OUT` up to 475 (~9.5s). `qn_load` reports this
   cleanly (`arena too small for t_out=...`), never corrupts memory.
2. **`MAX_CYCLES`** (`sim/verilator_qn/sim_main_qn_rtl.cpp`) bounds how many
   cycles the Verilator harness will simulate before giving up. Cycle count
   scales roughly linearly with `T_OUT` (~7.9M cycles/output frame,
   measured: `T_OUT=70` -> 554.5M real RTL cycles) -- raised to 6 billion,
   covering the full PSRAM-bound range (`T_OUT=475` needs an estimated
   ~3.76B cycles) with headroom. If you ever see `[sim] TIMEOUT after N
   cycles` with **no** `transcript(...)` line, this is the limit that bit
   you -- it fails silently different from the PSRAM check: the run just
   runs out of budget partway through and reports nothing.

If you need longer than ~9.5s: raise `PSRAM_BYTES` (e.g.
`PSRAM_BYTES=4194304` covers up to ~35s) **and** raise `MAX_CYCLES`
proportionally (~7.9M x your new max `T_OUT`, with margin) in
`sim_main_qn_rtl.cpp`, then run `make -C sim/verilator_qn clean` before
rebuilding -- the RTL sim binary is only rebuilt when `CONFIG` changes, not
when `PSRAM_BYTES`/`QSPI_BYTES` change, so a stale binary would silently
keep the old (too-small) memory size even after editing the Makefile.

---

## Checking accuracy at scale (not single clips)

The single-clip commands above are for "does this work" testing. For real
accuracy numbers (WER against LibriSpeech), see the gate scripts in
`sw/tinyml_reference/quartznet_run_{fp32,int8_ort,firmware}.py` and their
writeups in `docs/07e`-`07i`. Headline measured numbers: FP32 WER 4.44%,
int8-ORT WER 4.50-4.57%, real-firmware WER 4.50% (exactly matching int8-ORT
on test-clean) -- all against LibriSpeech dev-clean/test-clean, all
documented with the exact commands to reproduce them.

---

## Everything else, one level up

- `docs/07_quartznet_pivot.md` -- the original design doc (why QuartzNet,
  the architecture, the work-plan stages A-F).
- `docs/07a_sram_macro_notes.md` -- fakeram45 macro density/orientation notes.
- `docs/07d_soc_integration_and_gap2_start.md` onward -- the actual build
  log, RTL↔SoC integration, real-weights pipeline (one doc per gate: `07e`
  FP32 baseline, `07f` int8 calibration, `07g` export bridge, `07h`
  end-to-end firmware, `07i` calibration ablation, `07j` combined milestone).
- `CLAUDE.md`'s Stage 7 status block -- the single most up-to-date summary
  of what's done, with every measured number and every bug found along the
  way.

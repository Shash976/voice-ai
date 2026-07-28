# fakeram45 SRAM macro integration on nangate45 — verified reference

This repo was 100% flip-flop-based until Stage 7. The QuartzNet accelerator
needs a real on-chip memory system, so macro integration was de-risked with a
minimal spike (`rtl/spike_sram/sram_spike.v`) *before* writing the real RTL.

**Result: it works end-to-end.** Full ORFS flow to `6_final.gds`, clean.

## Verified result

| metric | value |
|---|---|
| macros placed | 4 — 2× `fakeram45_512x64` + 2× `fakeram45_1024x32` (16 KB total) |
| `6_final.gds` | produced, 2,203,244 B |
| detailed-route DRC | **0 violations** |
| timing | **WNS 0.00 ns, TNS 0.00 ns** (clean) |
| design area | 68,770 µm² @ 40% utilization |
| chip area (synth) | 68,188 µm², of which sequential 352.7 µm² (**0.52%**) |

That last row is the headline: with only 16 KB of SRAM, standard cells are
already noise. It confirms empirically that the real design's area will be
almost entirely macro area.

## What the design config must set

```make
export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram45_512x64.lef \
                         $(PLATFORM_DIR)/lef/fakeram45_1024x32.lef
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/fakeram45_512x64.lib \
                         $(PLATFORM_DIR)/lib/fakeram45_1024x32.lib
export DIE_AREA  = 0 0 460 410
export CORE_AREA = 10.07 11.2 450 400
export MACRO_PLACE_HALO = 10 10
```

### Ordering constraint — the likeliest silent failure
`LIB_FILES` is composed **inside the platform config**, which `variables.mk:40`
includes *after* the design config. So `ADDITIONAL_LIBS` must be set in the
design `config.mk`. Set it anywhere later and it silently never reaches
`LIB_FILES` — yosys then cannot see the macro ports, with a confusing downstream
error rather than an obvious one.

### Explicit DIE_AREA / CORE_AREA, not CORE_UTILIZATION
Utilization-derived sizing is fragile once macros dominate the area. Give the
floorplan explicit numbers.

## What you do *not* need

- **`GDS_ALLOW_EMPTY`** — already set in the platform config as
  `GDS_ALLOW_EMPTY ?= fakeram.*` (`platforms/nangate45/config.mk:91`; asap7 has
  the equivalent at :217). `def2stream.py:82-96` only errors on empty LEF cells
  that don't match that regex. Do not set it in the design config.
- **A Verilog blackbox declaration** — `synth_stdcells.tcl` does
  `read_liberty -lib {*}$::env(LIB_FILES)`, so yosys learns the port list from
  the `.lib`. Upstream `ariane133/macros.v` likewise declares only a wrapper,
  never `module fakeram45_256x16`.
- **A `PDN_TCL` override** — `grid_strategy-M1-M4-M7.tcl` already defines
  `CORE_macro_grid_1/2` with metal5/metal6 straps and metal4↔metal5 connects,
  matching the fakeram45 metal4 VDD/VSS stripes. Confirmed in the log: a grid
  was inserted for all four macro instances automatically.

## Macro selection: smaller is denser (counterintuitive)

Computed from the `.lib` areas across all 22 fakeram45 geometries:

| macro | area µm² | capacity | **µm²/byte** |
|---|---|---|---|
| **`fakeram45_1024x32`** | 16,406 | 4 KB | **4.005** ← best |
| `fakeram45_512x64` | 17,301 | 4 KB | 4.224 |
| `fakeram45_256x96` | 13,702 | 3 KB | 4.460 |
| `fakeram45_2048x39` | 45,479 | 9.75 KB | 4.555 |
| `fakeram45_128x256` | 33,990 | 4 KB | 8.298 |

The 9.75 KB macro is **12.1% worse per byte** than the 4 KB one, inverting
normal SRAM-compiler intuition. It is an artifact of fakeram45 being a synthetic
generator, but it is ground truth for this flow. Building 69 KB from `2048x39`
would cost ~68,500 µm² more than from `1024x32` — about 3.5× the entire current
accelerator, purely from picking the "obvious" big macro.

**Banking recommendation:** `1024x32` is densest but presents only a 32-bit port
(4 int8/cycle). Pair **two `1024x32` side-by-side as one logical 64-bit port** —
best density *and* a 64-bit port, at the cost of one extra instance to place.

## Physical gotchas for the real (~18-macro) floorplan

- **All signal pins are on the west edge only** (metal3, x≈0…0.070).
- **OBS blankets metal1/2/3** across the full macro footprint — only metal4+
  routes over a macro.

Together these make macro **orientation a first-order routability decision**. At
4 macros the router had no trouble (0 DRC), but 4 macros is too few to expose
the problem — an 18-macro floorplan with every macro's pins facing the same wall
is a congestion hazard. Test orientation variants before committing to a
floorplan; three candidate `MACRO_PLACEMENT_TCL` variants are staged in
`physical/orfs/make/nangate45/sram_spike/` for that experiment.

## Reproducing

```bash
bash physical/orfs/make/run_spike.sh nangate45
```

Outputs land in `physical/orfs/runs/{results,reports,logs}/nangate45/sram_spike/base/`.

## Caveats

- fakeram45 macros have **no behavioural model and no GDS**. They are
  placement/timing/area placeholders. Functional simulation of the memory must
  use a separate behavioural model — the same RTL-core / C++-testbench split the
  repo already uses for the 256 KB main memory.
- Control-pin polarity (`ce_in`/`we_in` active low) follows upstream
  `ariane133/macros.v`. With no behavioural model this is unverifiable and
  physically irrelevant; matching upstream just avoids inventing a third
  convention.

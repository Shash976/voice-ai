# Machine setup — reproducing Stage 7 work on a new box

Everything in Stage 7 was developed **without sudo**. Nothing below needs root
except where explicitly noted.

## 1. Python (conda)

```bash
conda env create -f environment.yml
conda activate voiceai
```

Verify:

```bash
python sw/tinyml_reference/quartznet_budget.py
```

Expect `TOTAL weights (= MACs per output frame): 18,847,040`.

Then the full software chain — no hardware tools needed:

```bash
python sw/tinyml_reference/quartznet_topology.py     # topology + bandwidth model
python sw/tinyml_reference/quartznet_descriptors.py  # emits descriptor table + blob sizes
python sw/tinyml_reference/quartznet_ref.py          # NumPy golden model
```

`quartznet_ref.py` should end with `replay : 187/187 descriptors reproduce from
blobs`. It writes ~20 MB of blobs to `build/quartznet/` (gitignored, and
regenerable in seconds from the seed — do not commit them).

## 2. Verilator

Used by `rtl/tb` (unit TB) and `sim/verilator` (full PicoRV32 sim). Developed
against **v5.048**. A local (non-root) build works fine:

```bash
git clone https://github.com/verilator/verilator && cd verilator
autoconf && ./configure --prefix=$HOME/local && make -j$(nproc) && make install
export PATH=$HOME/local/bin:$PATH
```

Verify the unit TB across the full parameter matrix:

```bash
cd rtl/tb && for L in 1 2 4 8 16 32; do for A in 24 32; do make clean >/dev/null && make LANES=$L ACC_W=$A | tail -1; done; done
```

Expect 12× `==== PASS : 0 mismatch(es) ====`.

## 3. OpenROAD-flow-scripts

Was at `/opt/OpenROAD-flow-scripts` (bundles its own Yosys 0.64 and OpenROAD).
Every script here honours `ORFS_DIR`, so a different location is fine:

```bash
export ORFS_DIR=/path/to/OpenROAD-flow-scripts
./physical/orfs/make/run.sh nangate45          # accelerator
./physical/orfs/make/run_spike.sh nangate45    # SRAM macro spike
```

Outputs go to `physical/orfs/runs/` (gitignored — it reached **885 MB** on the
dev machine, so never commit it). The curated text reports backing the numbers
in `docs/07_quartznet_pivot.md` are committed under `physical/orfs/measured/`.

> **Yosys version matters.** Yosys 0.64 asserts at `genrtlil.cc:2214` on
> signed/unsigned mixing. Verilator lint and Yosys 0.9 both *miss* these, so
> only a real ORFS synthesis run catches them. Do not assume RTL is clean
> because the TB passes.

### Known portability bug
`physical/orfs/synth_area.sh` hardcodes `$HOME/OpenROAD-flow-scripts` and a bare
`yosys`, neither of which resolved on the dev machine. Not yet fixed. Use
`run.sh` (which honours `ORFS_DIR`) or invoke Yosys directly meanwhile.

## 4. RISC-V toolchain — ⚠️ still missing

The firmware Makefiles want `riscv64-linux-gnu-gcc` (`CROSS ?= riscv64-linux-gnu`).
It was **not available on the dev machine and could not be apt-installed** (no
sudo), so the full-system PicoRV32 sim was never run for Stage 7.

**Consequence:** the `sim/verilator/sim_main.cpp` latency change (+1 drain cycle
per operation, from the requantize pipelining) is **verified against the RTL by
`rtl/tb`, but not verified end-to-end.** It is flagged in a comment at the change
site. Please confirm it on the first full run on a machine that has the toolchain.

Options, easiest first:

```bash
sudo apt install gcc-riscv64-linux-gnu          # if you have root
```

Without root, use the xPack prebuilt tarball (no install step, just untar and
put it on `PATH`) from
`https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases`, then set
`CROSS` accordingly, e.g.:

```bash
make -C firmware/picorv32_baremetal CROSS=riscv-none-elf
```

Note the toolchain is only needed for firmware and the full-system sim. The unit
TB, the ORFS flows, and the entire Python chain do not need it.

## What is intentionally not committed

| path | size | why |
|---|---|---|
| `physical/orfs/runs/` | 885 MB | regenerable; GDS/ODB/logs |
| `build/quartznet/` | 24 MB | regenerable in seconds from the seed |
| `physical/orfs/make/src/` | 40 KB | staged RTL copies, rewritten every run |

Committed instead: `physical/orfs/measured/` (240 KB of text reports) — the
evidence for the timing/area numbers quoted in the docs.

# Onboarding Docs

New here? Read **[00_onboarding_overview.md](00_onboarding_overview.md)** first — it's
the map. Then follow the numbered docs in order.

| # | Doc | Covers |
|---|-----|--------|
| 00 | [Onboarding Overview](00_onboarding_overview.md) | Big picture, the 6 stages, the Windows/WSL split, glossary. **Start here.** |
| 01 | [Model & Quantization](01_model_and_quantization.md) | Stage 1–2: TinyVAD architecture, int8 quantization, the C inference engine. |
| 02 | [Firmware & Simulation](02_firmware_and_simulation.md) | Stage 3: PicoRV32, bare-metal firmware, Verilator, memory map, build flags. |
| 03 | [The MAC Accelerator](03_accelerator.md) | Stage 4: the memory-mapped accelerator, register map, hooks, driver. |
| 05 | [Commands Cheat Sheet](05_commands_cheatsheet.md) | Every command, which machine, expected output, common pitfalls. |
| 06 | [RTL to GDS](06_rtl_to_gds.md) | Stage 6: synthesizable accelerator RTL, the Verilator correctness gate, ORFS sky130hd/asap7 flow, physical metrics. |
| — | Stage 5 — Design-space optimization | Extracted to the standalone **[eda-rl](https://github.com/Shash976/eda-rl)** repo: a design-agnostic multi-fidelity funnel optimizer over the ORFS flow. |

These docs were written against the actual code (June 2026) and are more precise than
the top-level [`../README.md`](../README.md) where they differ. The authoritative
long-form plan is [`../pocket_ai_voice_recorder_riscv_tinyml_plan.md`](../pocket_ai_voice_recorder_riscv_tinyml_plan.md).

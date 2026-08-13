# constraint.sdc — timing constraints for quartznet_accel (nangate45).
#
# Same shape as nangate45/tinymac_accel/constraint.sdc: one clock on the `clk`
# port, every other port I/O-delayed at a fixed fraction of the period.
#
# WHY 10.0 ns AND NOT tinymac's 2.0 ns. tinymac's 269 MHz / 415 MHz numbers do
# not transfer. Stage 7 pipelined requantize, which moved the critical path off
# the Q31 multiply and onto `i_in_chunk -> acc` — a path that IS LANES-dependent
# (CLAUDE.md Stage-7 item (a)) — and this design runs at LANES=32, eight times
# tinymac's 4. Nobody has measured LANES=32 Fmax.
#
# Measured rather than guessed: a pre-layout OpenSTA run on the Yosys-0.64 mapped
# netlist (no wire RC, no clock-tree skew) gives 6.401 ns worst register-to-
# register and 5.181 ns worst output arrival, i.e. ~6.5 ns min period. 10.0 ns
# leaves ~35% for interconnect and skew, so a first run should close cleanly and
# any failure is a flow problem rather than a timing problem — which is the whole
# point of the first run.
#
# To find real Fmax, walk the period down and re-read WNS, exactly as
# physical/orfs/measured/README.md documents: period_min = clk_period - wns.
# Suggested ladder after a clean 10.0 ns run: 8.0 -> 7.0 -> 6.5.
current_design quartznet_accel

set clk_name      core_clock
set clk_port_name clk
set clk_period    10.0
set clk_io_pct    0.2

set clk_port [get_ports $clk_port_name]
create_clock -name $clk_name -period $clk_period $clk_port

# Every non-clock input is constrained, INCLUDING the bd_* group. Those are real
# top-level ports of quartznet_accel (:185-190), not just of ext_mem_if: a real
# SoC ties bd_en=0 and drives memory through the MMIO MEM_* port, but tying them
# off here would let synthesis delete the bd_en mux at quartznet_accel.v:406-410
# and understate area for a decision that has not been made. Likewise the 11
# o_* tile-walk observability outputs stay constrained — o_new0[*] is in fact the
# worst I/O path in the pre-layout STA.
set non_clock_inputs [all_inputs -no_clocks]
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]

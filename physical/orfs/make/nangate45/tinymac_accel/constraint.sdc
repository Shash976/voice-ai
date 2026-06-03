# constraint.sdc — timing constraints for tinymac_accel (nangate45).
# The single clock is the `clk` port; rst_n and all data ports are I/O-delayed
# as a fraction of the period. Start relaxed at 2.0 ns (500 MHz) — the critical
# path is the 64-bit Q31 requantize multiply. Tighten toward the Stage-5 target
# and re-read WNS to find Fmax.
current_design tinymac_accel

set clk_name      core_clock
set clk_port_name clk
set clk_period    2.0
set clk_io_pct    0.2

set clk_port [get_ports $clk_port_name]
create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]

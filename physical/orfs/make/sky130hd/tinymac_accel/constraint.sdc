# constraint.sdc — timing constraints for tinymac_accel (sky130hd, 130 nm).
# 130 nm is slow; start at 10 ns (100 MHz) and tighten to find Fmax.
current_design tinymac_accel

set clk_name      core_clock
set clk_port_name clk
set clk_period    10.0
set clk_io_pct    0.2

set clk_port [get_ports $clk_port_name]
create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]

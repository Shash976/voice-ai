# constraint.sdc — timing constraints for tinymac_accel (asap7, 7 nm).
#
# asap7 is ~6-8x faster than nangate45 for the same logic, and the critical path
# here is the 64-bit Q31 requantize multiply (LANES-independent, same as the
# other PDKs). nangate45 closed at Fmax ~269 MHz (3.72 ns); on 7 nm expect the
# same path to land in the GHz range. Start at 1.0 ns (1 GHz) as an aggressive
# first target, run the flow, then read the reported period_min / fmax from
# reports/asap7/tinymac_accel/base/6_finish.rpt and re-tighten (or relax) here.
current_design tinymac_accel

set clk_name      core_clock
set clk_port_name clk
set clk_period    1.0
set clk_io_pct    0.2

set clk_port [get_ports $clk_port_name]
create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]

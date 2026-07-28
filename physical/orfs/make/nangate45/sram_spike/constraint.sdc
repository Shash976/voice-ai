# constraint.sdc — timing constraints for sram_spike (nangate45).
#
# Same shape as nangate45/tinymac_accel/constraint.sdc: one clock on `clk`,
# all other ports I/O-delayed as a fraction of the period.
#
# 2.0 ns (500 MHz) is a deliberately optimistic starting point.  The dominant
# path here is fakeram45 clk -> rd_out through the XOR fold into the dout
# register, which is a macro-internal delay this design cannot influence — so
# whatever WNS comes back is a property of the macro, not of the logic, and is
# exactly the number worth recording for the real build.
current_design sram_spike

set clk_name      core_clock
set clk_port_name clk
set clk_period    2.0
set clk_io_pct    0.2

set clk_port [get_ports $clk_port_name]
create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]

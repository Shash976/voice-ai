# tinymac_accel timing constraints.
# Bring-up clock for sky130hd sanity = 100 MHz (10 ns). The Stage-5 design
# target is 5 ns (200 MHz); slack at that target is reported on asap7.
set clk_name      core_clock
set clk_port_name clk
set clk_period    10000
set clk_io_pct    0.2

set clk_port [get_ports $clk_port_name]
create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]

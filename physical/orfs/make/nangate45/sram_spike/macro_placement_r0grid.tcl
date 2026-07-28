# macro_placement_r0grid.tcl — Phase-2 variant: ALL MACROS AT R0.
#
# The deliberately naive arrangement.  fakeram45 puts every signal pin on the
# macro's WEST edge (metal3 only, x = 0.000..0.070) and blankets metal1-3 with
# OBS, so at R0 all four macros present their pin wall to the west.  The right
# column's pins then face the *body* of the left column, which is opaque below
# metal4.
#
# Positions are byte-identical to the automatic (base) run so that ORIENTATION
# is the only variable in the comparison.
place_macro -macro_name {g_ram512x64\[0\].u_ram}  -location {20.07 154.665}  -orientation R0
place_macro -macro_name {g_ram512x64\[1\].u_ram}  -location {20.07 21.245}   -orientation R0
place_macro -macro_name {g_ram1024x32\[0\].u_ram} -location {287.73 149.065} -orientation R0
place_macro -macro_name {g_ram1024x32\[1\].u_ram} -location {287.73 21.245}  -orientation R0

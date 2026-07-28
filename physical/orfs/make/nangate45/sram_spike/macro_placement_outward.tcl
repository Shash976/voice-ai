# macro_placement_outward.tcl — Phase-2 variant: PINS FACE THE DIE EDGES.
#
# The deliberate worst case, and the counterpart to `mirror`.  Left column at
# R0 (pins on its WEST edge, facing the die's west margin), right column at
# R180 (pins on its EAST edge, facing the die's east margin).  No macro pin
# faces the central channel where the standard cells are placed, so every
# macro net has to route around a 152 um opaque block (metal1-3 OBS) to get
# from its pin to the logic.
#
# Positions are byte-identical to the automatic (base) run.
place_macro -macro_name {g_ram512x64\[0\].u_ram}  -location {20.07 154.665}  -orientation R0
place_macro -macro_name {g_ram512x64\[1\].u_ram}  -location {20.07 21.245}   -orientation R0
place_macro -macro_name {g_ram1024x32\[0\].u_ram} -location {287.73 149.065} -orientation R180
place_macro -macro_name {g_ram1024x32\[1\].u_ram} -location {287.73 21.245}  -orientation R180

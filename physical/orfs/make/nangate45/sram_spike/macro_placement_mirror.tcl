# macro_placement_mirror.tcl — Phase-2 variant: PINS FACE A CENTRAL CHANNEL.
#
# Left column rotated 180 (pins move to its EAST edge), right column mirrored
# about X (pins stay on its WEST edge).  Both pin walls therefore front the
# ~115 um channel between the columns, where the standard cells sit.
#
# This is also, independently, what rtl_macro_placer chose unprompted in the
# base run — this file exists to reproduce that explicitly, so the comparison
# against r0grid isolates orientation from placer nondeterminism.
#
# Positions are byte-identical to the automatic (base) run.
place_macro -macro_name {g_ram512x64\[0\].u_ram}  -location {20.07 154.665}  -orientation R180
place_macro -macro_name {g_ram512x64\[1\].u_ram}  -location {20.07 21.245}   -orientation R180
place_macro -macro_name {g_ram1024x32\[0\].u_ram} -location {287.73 149.065} -orientation MX
place_macro -macro_name {g_ram1024x32\[1\].u_ram} -location {287.73 21.245}  -orientation MX

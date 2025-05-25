source /data/PDK/TSMC/ADFP16/PDK/Collaterals/scripts/setup.tcl
source prop.tcl

set_db max_cpus_per_server 8
set_db init_hdl_search_path ../src
# if you have child modules, you need to add the paths of the other .lib files
set_db library [list $slow_lib]
set_db interconnect_mode wireload

# read in sources
read_hdl -language sv src/$toplevel.v

# elaborate
elaborate $toplevel

set ghz 4
# constraints
set clock_period [expr 1.0/$ghz]
create_clock -name clk -period $clock_period clk

# these values are in ns, but you can change them to whatever you want.
# In fact, it may be imperative to change these depending on how you
# chain together your child modules
set_input_delay -clock clk -min -max 0 data_i
set_output_delay -clock clk -min -max 0 data_o

# uncomment this if you want retiming
# set_db design:$toplevel .retime true
set_db .syn_global_effort high
set_db .syn_map_effort high
set_db .syn_opt_effort high
set_db .syn_generic_effort high

syn_generic
syn_map
syn_opt

report_timing -nworst 5 > reports/$toplevel.timing.rpt
report_area > reports/$toplevel.area.rpt
write_hdl > reports/$toplevel.netlist.v

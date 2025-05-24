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

# constraints
set clock_period 0.5
create_clock -name clk -period $clock_period clock

# these values are in ns, but you can change them to whatever you want.
# In fact, it may be imperative to change these depending on how you
# chain together your child modules
set_input_delay -clock -min -max clk 0.0 data_i
set_output_delay -clock -min -max clk 0.0 data_o

# uncomment this if you want retiming
# set_db design:$toplevel .retime true

syn_generic
syn_map
syn_opt

report_timing -nworst 5 > reports/$toplevel.rpt 

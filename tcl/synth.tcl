set impl_sel "solution_0"

set PRJ_NAME Network

delete_project ${PRJ_NAME}_ip
open_project ${PRJ_NAME}_ip
set_top ${PRJ_NAME}

open_solution solution_1
set_part {xczu3eg-sbva484-1-i}

add_files src/${PRJ_NAME}.cpp
add_files -tb tb/${PRJ_NAME}Tb.cpp

create_clock -period 5

# Done to avoid forced pipelining of loops that could waste resources
# config_compile -pipeline_loops 200000
# csim_design

csynth_design

export_design

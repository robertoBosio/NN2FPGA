set impl_sel "solution_0"

set PRJ_NAME ${TOP_NAME}

delete_project ${PRJ_NAME}_ip
open_project ${PRJ_NAME}_ip
set_top ${PRJ_NAME}

open_solution solution_1
set_part {xczu3eg-sbva484-1-i}

add_files src/memory_management.cc
add_files src/${PRJ_NAME}.cc
add_files -tb tb/${DATASET}/${DATASET}_tb.cc

csim_design

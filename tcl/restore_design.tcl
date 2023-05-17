set impl_sel "solution_0"

set BOARD $::env(BOARD)

set TOP_NAME Network
set PRJ_NAME ${TOP_NAME}${BOARD}

open_project ${PRJ_NAME}_ip
open_solution solution_1

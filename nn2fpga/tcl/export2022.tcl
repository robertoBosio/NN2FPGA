# Grep firt element of args list
set PROJ [lindex $argv 3]

open_project ${PROJ}
open_solution solution
export_design
exit
import subprocess
import os
import pytest

PROJECT_NAME = "proj_unit_test"
FILE_DIR = f"/workspace/NN2FPGA/nn2fpga/library/include/utils"
TEST_DIR = "/workspace/NN2FPGA/nn2fpga/library/test/utils"
FILENAME = "CSDFG_utils"

def test_CSDFG_utils_simulation():

    # Write the Tcl script to a temporary file
    tcl_script = f"""
open_project -reset "{PROJECT_NAME}"
open_solution -reset solution0
add_files {FILE_DIR}/{FILENAME}.hpp
add_files -tb {TEST_DIR}/Unit{FILENAME}.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include"
csim_design
exit
"""

    tcl_file = "script.tcl"
    with open(tcl_file, 'w') as f:
        f.write(tcl_script)

    result = subprocess.run(
        ["vitis_hls", "-f", tcl_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    assert result.returncode == 0, f"Simulation failed: {result.stderr}"
    assert "passed" in result.stdout.lower(), f"Test did not pass: {result.stdout}"

    # Clean up the temporary Tcl file
    os.remove(tcl_file)

    # Clean up the project directory
    if os.path.exists(PROJECT_NAME):
        os.system(f"rm -rf {PROJECT_NAME}")

    # Clean up vitis_hls log files
    log_files = [f for f in os.listdir('.') if f.startswith('vitis_hls') and f.endswith('.log')]
    for log_file in log_files:
        os.remove(log_file) 
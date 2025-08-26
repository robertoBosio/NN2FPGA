import subprocess
import os
import pytest
import csnake

PROJECT_NAME = "proj_unit_test"
FILE_DIR = f"/workspace/NN2FPGA/nn2fpga/library"
FILENAME = "NHWCToStream"

def generate_config_file(config_dict):

    # Dump the tensors in a hpp file
    cwr = csnake.CodeWriter()
    cwr.include("<cstdint>")
    cwr.include("<array>")
    cwr.include("<ap_int.h>")
    cwr.include("DequantQuant.hpp")
    cwr.add_line("namespace test_config {")
    cwr.indent()
    for key, value in config_dict.items():
        if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
            cwr.add_line(f"const float {key} = {value}f;")
        else:   
            cwr.add_line(f"const int {key} = {value};")
    cwr.add_line(
        f"typedef DequantQuantPo2<0, {config_dict['OUT_DATAWIDTH']}, {config_dict['OUT_DATAWIDTH']}> Quantizer;"
    )
    cwr.add_line(f"typedef ap_int<{config_dict['AXI_DATAWIDTH']}> TInput;")
    cwr.add_line(f"typedef ap_int<{config_dict['OUT_DATAWIDTH']}> TOutput;")
    cwr.add_line(
        f"typedef ap_axiu<{config_dict['AXI_DATAWIDTH']}, 0, 0, 0> TInputWord;"
    )
    cwr.dedent()
    cwr.add_line("}")
    return cwr.code

def generate_hls_script(testconfig_file):
    return f"""
open_project "{PROJECT_NAME}"
open_solution -reset solution0
add_files {FILE_DIR}/include/{FILENAME}.hpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include"
add_files -tb {FILE_DIR}/test/Unit{FILENAME}.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include"
add_files -tb {testconfig_file}

csim_design
exit
"""

def runhls(tcl_file):
    # Write the Tcl script to a temporary file

    return subprocess.run(
        ["vitis_hls", "-f", tcl_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def run(config_dict):
    testconfig_file = "test_config.hpp"
    with open(testconfig_file, "w") as f:
        f.write(generate_config_file(config_dict))

    tcl_file = "script.tcl"
    with open(tcl_file, 'w') as f:
        f.write(generate_hls_script(testconfig_file))

    result = runhls(tcl_file)
    assert result.returncode == 0, f"Simulation failed: {result.stderr}"
    assert "passed" in result.stdout.lower(), f"Test did not pass: {result.stdout}"

    # Clean up the temporary Tcl file
    os.remove(tcl_file)
    os.remove(testconfig_file)

    # Clean up the project directory
    if os.path.exists(PROJECT_NAME):
        os.system(f"rm -rf {PROJECT_NAME}")

    # Clean up vitis_hls log files
    log_files = [f for f in os.listdir('.') if f.startswith('vitis_hls') and f.endswith('.log')]
    for log_file in log_files:
        os.remove(log_file)


def test_axi128_par2():
    config_dict = {
        "AXI_DATAWIDTH": 128,
        "OUT_DATAWIDTH": 8,
        "WIDTH": 4,
        "HEIGHT": 4,
        "CH": 4,
        "OUT_W_PAR": 1,
        "OUT_CH_PAR": 2,
        "DATA_PER_WORD": 16,
        "PIPELINE_DEPTH": 4,
    }
    run(config_dict)

def test_axi64_par6():
    config_dict = {
        "AXI_DATAWIDTH": 64,
        "OUT_DATAWIDTH": 8,
        "WIDTH": 4,
        "HEIGHT": 4,
        "CH": 3,
        "OUT_W_PAR": 2,
        "OUT_CH_PAR": 3,
        "DATA_PER_WORD": 8,
        "PIPELINE_DEPTH": 1,
    }
    run(config_dict)

def test_axi64_par3():
    config_dict = {
        "AXI_DATAWIDTH": 64,
        "OUT_DATAWIDTH": 8,
        "WIDTH": 4,
        "HEIGHT": 4,
        "CH": 3,
        "OUT_W_PAR": 1,
        "OUT_CH_PAR": 3,
        "DATA_PER_WORD": 8,
        "PIPELINE_DEPTH": 2,
    }
    run(config_dict)

import subprocess
import os
import pytest

PROJECT_NAME = "project_mobilenetv2_a8w8b16_test"
ONNX_NAME = "mobilenet_v2_a8w8b16_71.734.onnx"
PROJECT_DIR = f"../work/{PROJECT_NAME}"

@pytest.fixture(scope='module', autouse=True)
def cleanup():
    passed_tests = []
    yield passed_tests

    # Teardown code after all tests in the module run
    if os.path.exists(PROJECT_DIR):
        os.system(f"rm -rf {PROJECT_DIR}")

def test_mobilenet_v2_a8w8b16_compilation():
    result = subprocess.run(
        [
            "make",
            "generate",
            "TOP_NAME=mobilenet_v2",
            f"ONNX_PATH=../models/onnx/{ONNX_NAME}",
            "DATASET=imagenet",
            "BOARD=ZCU102",
            "DYNAMIC_INIT=1",
            f"PRJ_ROOT={PROJECT_DIR}"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    assert result.returncode == 0, f"Conversion failed: {result.stderr}"
    assert "compilation successful." in result.stdout.lower()

def test_mobilenet_v2_a8w8b16_csim_inference():
    result = subprocess.run(
        [
            "make",
            "csim",
            "TOP_NAME=mobilenet_v2",
            f"ONNX_PATH=../models/onnx/{ONNX_NAME}",
            "DATASET=imagenet",
            "BOARD=ZCU102",
            "DYNAMIC_INIT=1",
            f"PRJ_ROOT={PROJECT_DIR}"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    assert result.returncode == 0, f"Inference failed: {result.stderr}"
    assert "test passed" in result.stdout.lower()

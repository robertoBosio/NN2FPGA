import subprocess
import os
import pytest

PROJECT_NAME = "project_resnet20_a8w8b16_test"
ONNX_NAME = "resnet20_a8w8b16.onnx"
PROJECT_DIR = f"../work/{PROJECT_NAME}"

@pytest.fixture(scope='module', autouse=True)
def cleanup():
    yield
    if os.path.exists(PROJECT_DIR):
        os.system(f"rm -rf {PROJECT_DIR}")

def test_resnet20_a8w8b16_compilation():
    result = subprocess.run(
        [
            "make", 
            "generate", 
            "TOP_NAME=resnet20", 
            f"ONNX_PATH=../models/onnx/{ONNX_NAME}", 
            "DATASET=cifar10", 
            "BOARD=KRIA", 
            "DYNAMIC_INIT=1", 
            f"PRJ_ROOT={PROJECT_DIR}"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    assert result.returncode == 0, f"Conversion failed: {result.stderr}"
    assert "compilation successful." in result.stdout.lower()

def test_resnet20_a8w8b16_csim_inference():
    result = subprocess.run(
        [
            "make", 
            "csim", 
            "TOP_NAME=resnet20", 
            f"ONNX_PATH=../models/onnx/{ONNX_NAME}", 
            "DATASET=cifar10", 
            "BOARD=KRIA", 
            "DYNAMIC_INIT=1", 
            f"PRJ_ROOT={PROJECT_DIR}"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    assert result.returncode == 0, f"Inference failed: {result.stderr}"
    assert "test passed" in result.stdout.lower()

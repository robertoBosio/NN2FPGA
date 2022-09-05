import os
import sys

def io_shape(io):
    input_shapes = [
        d.dim_value for d in io.type.tensor_type.shape.dim
    ] 

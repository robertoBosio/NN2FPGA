import os
import sys
import math

def find_divisors(layers_info, clamp=33):
    all_divisors = []
    layers_divisors = []
    layers_offset = []
    layers_name = []
    offset = 0
    for i, layer_info in enumerate(layers_info):
        divisors = 1
        layers_offset.append(offset)
        all_divisors.append(1)
        for k in range(2, min([layer_info[6], clamp])+1):
            if (layer_info[6] % k) == 0:
                all_divisors.append(k)
                divisors = divisors + 1
        layers_divisors.append(divisors)
        offset = offset + divisors
        layers_name.append(layer_info[0])
    return all_divisors, layers_divisors, layers_offset, layers_name

def find_range(divisors, ilp_value):
    low_range = 1
    high_range = 1
    for i, divisor in enumerate(divisors):
        if (i == 0):
            high_range = divisor
        
        if ilp_value >= divisor:
            low_range = divisor

        if ilp_value <= divisor:
            high_range = divisor
    
    return low_range, high_range
            
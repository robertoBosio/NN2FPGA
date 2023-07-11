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
    low_range = []
    high_range = []
    for i, divisor in enumerate(divisors):
        
        if ilp_value >= divisor:
            low_range.append(divisor)

        if ilp_value <= divisor:
            high_range.append(divisor)
    
    if len(low_range) == 0:
        low_range.append(divisors[0])
    if len(high_range) == 0:
        high_range.append(divisors[-1])

    return max(low_range), min(high_range)
            
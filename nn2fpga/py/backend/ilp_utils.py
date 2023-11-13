import os
import sys
import math

def find_divisors(layers_info):
    all_divisors = []
    layers_divisors = []
    layers_offset = []
    offset = 0
    for layer_info in layers_info:
        # FIX: adapting to not matching och for merged layers
        # compute maximum common submultiple between layer_info[6] and layer_info[7]
        # if layer_info[7] is present
        if layer_info["max_par"] == layer_info["max_par_1x1"]:
            if layer_info["max_par"] > layer_info["max_par_1x1"]:
                max_value = layer_info["max_par"]
                min_value = layer_info["max_par_1x1"]
            else:
                max_value = layer_info["max_par_1x1"]
                min_value = layer_info["max_par"]
            
            for k in range(2, min_value + 1):
                if (max_value % k) == 0 and (min_value % k) == 0:
                    max_par = k
        else:
            max_par = layer_info["max_par"]
            max_value = layer_info["max_par"]

        divisors = 1
        layers_offset.append(offset)
        all_divisors.append(1)
    
        for k in range(2, max_value + 1):
            if (max_par % k) == 0:
                all_divisors.append(k)
                divisors = divisors + 1
        layers_divisors.append(divisors)

        offset = offset + divisors
    return all_divisors, layers_divisors, layers_offset

def find_range(divisors, ilp_value):
    low_bound = divisors[0]
    low_index = 0
    high_bound = divisors[-1]
    high_index = len(divisors) - 1
    for i, divisor in enumerate(divisors):
        if (divisor >= ilp_value and abs(divisor - ilp_value) < abs(high_bound - ilp_value)):
            high_bound = divisor
            high_index = i
        if (divisor <= ilp_value and abs(divisor - ilp_value) < abs(low_bound - ilp_value)):
            low_bound = divisor
            low_index = i

    return low_bound, low_index, high_bound, high_index
            
def find_higher_mult(ref, high_mult):
    print("ref: ", ref)
    print("high_mult: ", high_mult)
    for i in range(high_mult, -1, -1):
        if ((ref % i) == 0) and ((high_mult % i) == 0):
        # if ((ref % i) == 0):
            return i
    
    assert (0 == 1)
    
def find_lower_mult(low_mult, ref):
    for i in range(low_mult, ref+1):
        if ((ref % i) == 0):
        # if ((ref % i) == 0):
            return i
    
    assert (0 == 1)
    
import os
import sys
import math

def find_divisors(layers_info):
    all_divisors = []
    layers_divisors = []
    layers_offset = []
    layers_name = []
    offset = 0
    for i, layer_info in enumerate(layers_info):
        # FIX: adapting to not matching och for merged layers
        # compute maximum common submultiple between layer_info[6] and layer_info[7]
        # if layer_info[7] is present
        if len(layer_info) > 7:
            if layer_info[6] > layer_info[7]:
                max_value = layer_info[6]
                min_value = layer_info[7]
            else:
                max_value = layer_info[7]
                min_value = layer_info[6]
            
            for k in range(2, min_value+1):
                if (max_value % k) == 0 and (min_value % k) == 0:
                    max_par = k
        else:
            max_par = layer_info[6]
            max_value = layer_info[6]

        divisors = 1
        layers_offset.append(offset)
        all_divisors.append(1)
        for k in range(2, max_par+1):
            if (max_par % k) == 0:
                all_divisors.append(k)
                divisors = divisors + 1
        for k in range(max_par+1, max_value+1):
            if (max_par % k) == 0:
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
    
import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np

def return_index(fh, fw, ow_ops, stride, iw, ih):
    """ Return the index of next element in the line buffer """
    base_index = ih * (fw + (ow_ops - 1) * stride) + iw

    # Case in which the index is still in the same row of the window
    if iw + ow_ops < (fw + (ow_ops - 1) * stride):
        return base_index + ow_ops
    elif (iw >= (fw - 1)) and (iw < ow_ops):
        return (ih + 1) * (fw + (ow_ops - 1) * stride) + iw
    else:
        return (ih + 1) * (fw + (ow_ops - 1) * stride) + iw % ow_ops
    print("ERROR: return_index")
    sys.exit(1)

def depth_simulator(dfh, dfw, ow_ops, stride, iw, ih, ich, ich_ops, pad):
    """ Compute the max depth for each pre_pad stream by simulating the number
      of inputs written in each stream before the conv starts """
    
    # Define the padded input dimensions
    ih_padded = ih + (pad * 2)
    iw_padded = iw + (pad * 2)

    # Create a window to simulate the convolution
    window = np.zeros((dfh, dfw))
    h_step = stride
    w_step = ow_ops * stride

    # Create the simulation tensor with data needed to start the first convolution
    tensor = np.zeros((dfh * 2 - 1, iw_padded), dtype=int)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            
            # Condition for the padding
            if not (i < pad or j < pad or i >= ih_padded - pad or j >= iw_padded - pad):
                
                # Condition for the useful data to start the first convolution
                # Space to do two final windows as we for the pad input we need 
                # all the streams to be not empty due to scheduling. Otherwise
                # the stream padded is empty.
                if not (i == dfh - 1 and j >= dfw):

                    if i < dfh:
                        tensor[i][j] = int(ich // ich_ops)
    
    # Simulate the convolution from the point of view of the shift_op
    print(f"ih_padded: {ih_padded}, iw_padded: {iw_padded}")
    print(f"dfh: {dfh}, dfw: {dfw}, stride: {stride}, ow_ops: {ow_ops}")
    for i in range(0, dfh, h_step):
        for j in range(0, iw, w_step):
            print(f"i: {i}, j: {j}")
            for fh in range(dfh):
                for fw in range(dfw):
                    # if (i + fh) < ih_padded and (j + fw) < iw_padded:
                    window[fh][fw] += tensor[i + fh][j + fw]
    
    window_max = window
    # Dynamic analysis of the depth
    # Writing in the tensor the data in the correct position produced for each step 
    # (not considering clock cycle accurate simulation, in this case one channel is produced all in parallel)
    for i_prod in range(dfh - pad - 1, ih):
        for j_prod in range(dfw - pad - 1, iw, ow_ops):
            tensor = np.zeros((ih_padded, iw_padded), dtype=int)
            for ow_prod in range(ow_ops):
                tensor[i + pad][j + pad + ow_prod] = int(ich // ich_ops)
           
            # shift_op
            for i in range(0, ih, h_step):
                for j in range(0, iw, w_step):
                    for fh in range(dfh):
                        for fw in range(dfw):
                            pass
                            # if (i + fh) < ih_padded and (j + fw) < iw_padded:
                            # window[fh][fw] += tensor[i + fh][j + fw]
    
    
    
    return window

def parse(name, node, debug=False):

    stride = node["stride"]
    ow_ops = node["ow_ops"]
    dfh = node["fh"]
    dfw = node["fw"] + ((ow_ops - 1) * stride)
    dindex = dfh * dfw
    line_buffer_blocks = []
    
    input_name  = node["input"][0]
    input_type_name = input_name.replace("_skip", "")

    output_name = input_name
    output_type_name = output_name.replace("_skip", "")

    #pad_value = int(node["pad"] * (node["fw"] - 1) / 2)
    pad_value = int(node["pad"])

    # window_depth = depth_simulator(dfh, dfw, ow_ops, stride, node["iw"], node["ih"], node["ich"], node["ich_ops"], node["pad"])
    
    for fh in range(dfh):
        for fw in range(dfw):
            
            index = fh * dfw + fw
            out_index = return_index(node["fh"], node["fw"], ow_ops, stride, fw, fh)
            if debug:
                print("index: %0d" % index, "out_index: %0d" % out_index, "fh: %0d" % fh, "fw: %0d" % fw)

            block = {}
            block["func"] = "shift_op"

            # Template parameters
            block["template"] = []
            if index < ow_ops:
                block["template"].append("t_%s_struct" % input_type_name)
            else:
                block["template"].append("t_%s_lb_struct" % input_type_name)
            block["template"].append("t_%s_lb_struct" % input_type_name)
            if out_index < (dindex):
                block["template"].append("t_%s_lb_struct" % input_type_name)
            else:
                block["template"].append("std::nullptr_t")
            block["template"].append("c_%s_ich" % name)
            block["template"].append("c_%s_och" % name)
            block["template"].append("c_%s_ih" % name)
            block["template"].append("c_%s_iw" % name)
            block["template"].append("c_%s_oh" % name)
            block["template"].append("c_%s_ow" % name)
            block["template"].append("c_%s_fh" % name)
            block["template"].append("c_%s_fw" % name)
            block["template"].append("c_%s_stride" % name)
            block["template"].append("c_%s_pad" % name)
            block["template"].append("%0d" % (dfh - 1 - fh))
            block["template"].append("%0d" % (dfw - 1 - fw))
            block["template"].append("c_%s_ow_ops" % name)
            if index < ow_ops:
                if node["adjust_line_buffer"]:
                    block["template"].append("%0d" % node["adjust_ops"])
                else:
                    block["template"].append("%0d" % node["in_ops"])
            else:
                block["template"].append("%0d" % node["line_ops"]) #line_ops
            block["template"].append("%0d" % node["line_ops"]) #line_ops

            block["args"] = []

            # dfw - 1 - fw gives us the position of the input in the inverted line buffer, applying 
            # the modulo we get the correct ow_ops stream moving that section of the tensor. pad_value
            # is used to adjust the position of the input in the line buffer.
            if index < ow_ops:
                block["args"].append("s_%s[%0d]" % (input_name, (dfw - 1 - fw - pad_value)%ow_ops))
            else:
                block["args"].append(
                    "s_%s_data[%0d]" % (input_name, index-ow_ops)
                )

            block["args"].append(
                "s_%s_pre_pad[%0d]" % (input_name, index)
            )

            if out_index < (dindex):
                block["args"].append(
                    "s_%s_data[%0d]" % (input_name, out_index-ow_ops)
                )
            else:
                block["args"].append(
                    "s_%s_null[%0d]" % (input_name, dindex-index-1)
                )

            block["declare"] = []

            if (index == 0):
                declare = {}
                declare["name"] = "s_%s_data" % output_name
                declare["type"] = "t_%s_lb_struct" % output_name
                declare["is_array"] = True
                declare["dim"] = dindex-ow_ops
                block["declare"].append(declare)

                declare = {}
                declare["name"] = "s_%s_null" % output_name
                declare["type"] = "std::nullptr_t"
                declare["is_array"] = True
                declare["dim"] = ow_ops
                block["declare"].append(declare)

                declare = {}
                declare["name"] = "s_%s_pre_pad" % output_name
                declare["type"] = "t_%s_lb_struct" % output_name
                declare["is_array"] = True
                declare["dim"] = dindex
                block["declare"].append(declare)

            block["pragma"] = []

            output_ratio = int((node["ih"]*node["iw"])/(node["oh"]*node["ow"]))
            if output_ratio > 1: 
                depth = node["och"]*output_ratio+1
                # impl = "BRAM"
                impl = "AUTO"
            else:
                depth = node["fh"]*node["fw"]+1
            # depth = 2
            impl = "AUTO"

            # print(f"Assinging {window_depth[fh][fw]} instead of {depth} to s_{output_name}_pre_pad[{index}]")
            # depth = int(window_depth[dfh - 1 - fh][dfw - 1 - fw]) + 1
            # pragma = {}
            # pragma["name"] = "stream"
            # options = [
            #     ["variable", f"s_{output_name}_pre_pad[{index}]"],
            #     ["depth", depth],
            #     # ["depth", "2"],
            #     ["type", "fifo"],
            # ]
            # pragma["options"] = options
            # block["pragma"].append(pragma)

            if (index == 0):
                pragma = {}
                pragma["name"] = "stream"
                options = [
                    ["variable", "s_%s_pre_pad" % output_name],
                    ["depth", depth],
                    # ["depth", "2"],
                    ["type", "fifo"],
                ]
                pragma["options"] = options
                block["pragma"].append(pragma)

                pragma = {}
                pragma["name"] = "bind_storage"
                options = [
                    ["variable", "s_%s_pre_pad" % output_name],
                    # ["depth", "2"],
                    ["impl", impl],
                    ["type", "fifo"],
                ]
                pragma["options"] = options
                block["pragma"].append(pragma)

            pragma = {}

            # Line buffer long branch must be split in ow_ops parts
            if (fw > (dfw-ow_ops-1)):
                depth = int(node["iw"]/node["ow_ops"])*int(node["ich"]/node["ich_ops"])
            else:
                depth = int(node["ich"]/node["ich_ops"])
            # print(f"Assigning depth: 2 instead of {depth} to s_{output_name}_data[{index}]")
            # depth = 2

            if (index < (dindex-ow_ops)):

                pragma["name"] = "stream"
                options = [
                    ["variable", "s_%s_data[%0d]" % (output_name, index)],
                    ["depth", depth],
                    ["type", "fifo"],
                ]
                pragma["options"] = options

                block["pragma"].append(pragma)

                if (fh == 0) and (fw == 0):
                    # TODO: Check correcteness of variable selection
                    pragma = {}
                    pragma["name"] = "bind_storage"
                    options = [
                        ["variable", "s_%s_data[0]" % output_name],
                        # ["depth", "2"],
                        # ["impl", "BRAM"],
                        ["impl", "AUTO"],
                        ["type", "fifo"],
                    ]
                    pragma["options"] = options
                    block["pragma"].append(pragma)


            line_buffer_blocks.append(block)
    
    return line_buffer_blocks


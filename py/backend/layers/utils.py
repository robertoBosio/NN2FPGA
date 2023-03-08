import os
import sys
import onnx
from onnx import numpy_helper
import numpy as np

def write_weights(weight_shape, weight_name):

    c_och    = getattr(weight_shape, 'dims')[0]
    c_ich    = getattr(weight_shape, 'dims')[1]
    if (len(getattr(weight_shape, 'dims')) > 2):
        c_ih     = getattr(weight_shape, 'dims')[2]
        c_iw     = getattr(weight_shape, 'dims')[3]
    else:
        c_ih     = 1
        c_iw     = 1

    # fd.write("\ttypedef ap_uint<8> t_%s_st;\n" % (weight_name))
    # fd.write("\ttypedef ap_uint<8> t_%s;\n" % (weight_name))
    # fd.write("\tconst int c_%s_och = %d;\n" % (weight_name, c_och))
    # fd.write("\tconst int c_%s_ich = %d;\n" % (weight_name, c_ich))
    # fd.write("\tconst int c_%s_ih  = %d;\n" % (weight_name, c_ih))
    # fd.write("\tconst int c_%s_iw  = %d;\n" % (weight_name, c_iw))
    weights = numpy_helper.to_array(
        weight_shape
    )

    # TODO: handle weights quantization
    last_weight = True
    for ih in range(c_ih):
        for iw in range(c_iw):
            fd.write("\tconst uint8_t c_%s_st_%0d[] = {\n" % (weight_name, ih*c_iw+iw))
            for och in range(weights.shape[0]):
                for ich in range(weights.shape[1]):
                    # fd.write("%.2f" % (weights[och][ich][ih][iw]))
                    weight_value = np.random.randint(0, 256)
                    fd.write("%0d" % (weight_value))
                    fd.write(", ")

            fd.write("0")

            fd.write("};\n")
            fd.write("\n")

def write_array_stream(fd, name, channels=None, c_split=None):
    
    # Each stream has a channel width equal to the number of channels of the 
    # output feature 
    if (c_split is not None):
        fd.write(
            "\thls::stream<t_%s> s_%s[c_%s_split];\n" % (
                name,
                name,
                name
            )
        )
    else:
        fd.write(
            "\thls::stream<t_%s> s_%s[c_%s_ih*c_%s_iw];\n" % (
                name, 
                name,
                name,
                name
            )
        )

    if (channels is None):
        fd.write(
            "\t#pragma HLS STREAM variable=s_%s depth=3 type=fifo\n" % (
                name
            )
        )
    else:
        fd.write(
            "\t#pragma HLS STREAM variable=s_%s depth=%s type=fifo\n" % (
                name,
                channels
            )
        )

def write_last_flags(fd, layers_allocated):
    fd.write(
        "\thls::stream<ap_uint<1>> s_last_split[%0d];\n" % (
            layers_allocated + 1
        )
    )
    fd.write(
        "\t#pragma HLS STREAM variable=s_last_split depth=10 type=fifo\n"
    )

    # fd.write("\tSplitStream<\n")
    # fd.write("\t\t%0d\n" % (layers_allocated + 1))
    # fd.write("\t>(\n")
    # fd.write("\t\ts_last,\n")
    # fd.write("\t\ts_last_split\n")
    # fd.write("\t);\n")

    fd.write("\n")


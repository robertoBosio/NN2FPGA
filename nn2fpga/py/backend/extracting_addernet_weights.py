import os
import sys
#import onnx
import qonnx
from onnx import numpy_helper
import numpy as np
import math
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation import infer_shapes
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.cleanup import cleanup_model

def pack_weights(
    values,
    bits=8,
    is_bias=False
):
    """ Pack the weights in a single array """

    if (not is_bias):
        values = np.swapaxes(values, 0, 1)
    values = values.flatten()

    if bits >= 8:
        bytes_num = int(bits / 8)
        new_values = np.zeros([values.shape[0] * bytes_num])

        for i in range(values.shape[0]):
            data = values[i]
            data_bytes = np.zeros([bytes_num])
            for j in range(bytes_num):
                data_byte = int(data) & 0xff
                data_bytes[j] = data_byte
                data = int(data // 256)

            # Changing MSB to LSB order to ease hw reconstruction
            # of the original value
            for j in range(bytes_num - 1, -1, -1):
                new_values[i * bytes_num + j] = data_bytes[bytes_num - 1 - j]
            # for j in range(0, bytes_num):
            #     new_values[i*bytes_num+j] = data_bytes[j]

        values = new_values
    
    return values


def mem_shape_calc(node, fh, fw):
    """ Compute the memory shape needed to store the weights """
    ich = node["ich"]
    och = node["och"]
    ops = node["ops"]
    ich_ops = node["ich_ops"]
    depth = node["depth"]

    ich_iter_ops = ich // ich_ops
    och_iter_ops = och // ops
    if depth:
        och_iter_ops = 1

    assert ich_iter_ops > 0
    assert och_iter_ops > 0

    return [fh * fw, ich_iter_ops * och_iter_ops, ich_ops * ops]

def parse_on_chip_weights(
    node_info,
    node_quant,
    pre_values
):

    # Transforming a filter of dimension [och][ich][ih][iw] into one of
    # dimension [iw * ih][(ich * och)/(och_ops * ich_ops)][och_ops * ich_ops] where ops is the number 2D convolutions
    # computed in parallel

    dich = node_info["ich"]
    dih  = node_info["fh"]
    diw  = node_info["fw"]
    doch = node_info["och"]
    dops = node_info["ops"]
    dich_ops = node_info["ich_ops"]

    if node_info["depth"]:
        doch = 1

    # scale_factor = 2 ** node_quant["scale_factor"]
    scale_factor = node_quant["scale_factor"]
    signed = node_quant["signed"]
    bits = node_quant["bits"]
    narrow = node_quant["narrow"]

    narrow_h = 0
    if not signed and narrow:
      narrow_h = 1
    limit_h = 2**(bits-signed) - 1 - narrow_h

    narrow_l = 0
    if signed and narrow:
      narrow_l = 1
    limit_l = -1 * signed * 2**(bits-signed) + narrow_l

    doch_ops = int(doch / dops)
    dich_iter_ops = int(dich / dich_ops)

    assert dich_iter_ops > 0
    assert doch_ops > 0

    values = np.zeros(
        mem_shape_calc(node_info, node_info["fh"], node_info["fw"])
    )

    # print(f"ops: {dops}, ich_ops: {dich_ops}, ich: {dich}, och: {doch}, depth: {node_info['depth']}")
    # print(f"shape: {values.shape}")

    # Reordering the weights based on the parallelism needed by the convolution
    for ich in range(dich_iter_ops):
        for och in range(doch_ops):
            off = och * dops
            for ich_ops in range(dich_ops):
                for ih in range(dih - 1, -1, -1):
                    for iw in range(diw - 1, -1, -1):
                        off_ich = ich * dich_ops + ich_ops
                        for ops in range(dops):
                            quant_value = pre_values[off + ops][off_ich][ih][iw]
                            print(f"quant_value: {quant_value} {scale_factor} {quant_value / scale_factor}")
                            quant_value = np.round(quant_value / scale_factor)
                            
                            if (limit_h < quant_value):
                                quant_value = limit_h

                            if (limit_l > quant_value):
                                quant_value = limit_l
                            
                            index = ih * diw + iw
                            ch = ich * doch_ops + och
                            ops_index = ich_ops * dops + ops
                            values[dih * diw - 1 - index][ch][ops_index] = quant_value
    
    return values


onnx_path = "/home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/test/onnx/adder2d_expanded.onnx"
onnx_model = ModelWrapper(onnx_path)
cleanup_model(onnx_model)
inferred_model = onnx_model.transform(infer_shapes.InferShapes())
inferred_model = inferred_model.transform(InferDataTypes())
inferred_model.set_tensor_datatype("global_in", DataType["UINT8"])
inferred_model = inferred_model.transform(infer_shapes.InferShapes())

conv_params_names = [
    "DequantizeLinear_0_param0",
    "QuantizeLinear_1_param0",
    "QuantizeLinear_2_param0",
    "QuantizeLinear_3_param0",
    "QuantizeLinear_4_param0",
    "QuantizeLinear_5_param0",
    "QuantizeLinear_6_param0",
    "QuantizeLinear_7_param0",
    "QuantizeLinear_9_param0",
    "QuantizeLinear_8_param0",
    "QuantizeLinear_10_param0",
    "QuantizeLinear_11_param0",
    "QuantizeLinear_12_param0",
    "QuantizeLinear_13_param0",
    "QuantizeLinear_14_param0",
    "QuantizeLinear_16_param0",
    "QuantizeLinear_15_param0",
    "QuantizeLinear_17_param0",
    "QuantizeLinear_18_param0",
    "QuantizeLinear_19_param0",
    "QuantizeLinear_20_param0",
    "DequantizeLinear_1_param0",
]

conv_scalefactor_names = [
    "DequantizeLinear_0_param1",
    "QuantizeLinear_1_param1",
    "QuantizeLinear_2_param1",
    "QuantizeLinear_3_param1",
    "QuantizeLinear_4_param1",
    "QuantizeLinear_5_param1",
    "QuantizeLinear_6_param1",
    "QuantizeLinear_7_param1",
    "QuantizeLinear_9_param1",
    "QuantizeLinear_8_param1",
    "QuantizeLinear_10_param1",
    "QuantizeLinear_11_param1",
    "QuantizeLinear_12_param1",
    "QuantizeLinear_13_param1",
    "QuantizeLinear_14_param1",
    "QuantizeLinear_16_param1",
    "QuantizeLinear_15_param1",
    "QuantizeLinear_17_param1",
    "QuantizeLinear_18_param1",
    "QuantizeLinear_19_param1",
    "QuantizeLinear_20_param1",
    "DequantizeLinear_1_param1",
]

conv_parallelism = [
    {"name": "Conv_0", "ich_ops": 1, "ops": 1, "ich": 3, "och": 16, "depth": False, "fh": 3, "fw": 3},
    
    {"name": "Adder2d_0", "ich_ops": 1, "ops": 2, "ich": 16, "och": 16, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_1", "ich_ops": 2, "ops": 2, "ich": 16, "och": 16, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_2", "ich_ops": 2, "ops": 2, "ich": 16, "och": 16, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_3", "ich_ops": 1, "ops": 2, "ich": 16, "och": 16, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_4", "ich_ops": 4, "ops": 2, "ich": 16, "och": 16, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_5", "ich_ops": 4, "ops": 2, "ich": 16, "och": 16, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_6", "ich_ops": 1, "ops": 2, "ich": 16, "och": 32, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_7", "ich_ops": 1, "ops": 2, "ich": 16, "och": 32, "depth": False, "fh": 1, "fw": 1},
    {"name": "Adder2d_8", "ich_ops": 2, "ops": 2, "ich": 32, "och": 32, "depth": False, "fh": 3, "fw": 3},
    
    {"name": "Adder2d_9", "ich_ops": 1, "ops": 2, "ich": 32, "och": 32, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_10", "ich_ops": 1, "ops": 2, "ich": 32, "och": 32, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_11", "ich_ops": 2, "ops": 2, "ich": 32, "och": 32, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_12", "ich_ops": 1, "ops": 2, "ich": 32, "och": 32, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_13", "ich_ops": 2, "ops": 2, "ich": 32, "och": 64, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_14", "ich_ops": 2, "ops": 2, "ich": 32, "och": 64, "depth": False, "fh": 1, "fw": 1},
    {"name": "Adder2d_15", "ich_ops": 4, "ops": 2, "ich": 64, "och": 64, "depth": False, "fh": 3, "fw": 3},
    
    {"name": "Adder2d_16", "ich_ops": 4, "ops": 2, "ich": 64, "och": 64, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_17", "ich_ops": 2, "ops": 2, "ich": 64, "och": 64, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_18", "ich_ops": 2, "ops": 2, "ich": 64, "och": 64, "depth": False, "fh": 3, "fw": 3},
    {"name": "Adder2d_19", "ich_ops": 2, "ops": 2, "ich": 64, "och": 64, "depth": False, "fh": 3, "fw": 3},
    
    {"name": "Conv_1", "ich_ops": 1, "ops": 2, "ich": 64, "och": 10, "depth": False, "fh": 1, "fw": 1},
]

quant_nodes = []
init_info = {}

for info in inferred_model.graph.initializer:
    info_name = info.name.replace(".", "_")
    init_info[info_name] = info

values = numpy_helper.to_array(init_info["DequantizeLinear_0_param0"]) 
if values.ndim == 2:
  values = np.expand_dims(values, axis=-1)
  values = np.expand_dims(values, axis=-1)

tot_params = 0
for name in conv_params_names:
    values = numpy_helper.to_array(init_info[name])
    flatten_values = values.flatten()
    tot_params += len(flatten_values)

print(f"Total params: {tot_params}")
params = []
concat_weights = None
for index, name in enumerate(conv_scalefactor_names):
    name_param = conv_params_names[index]
    conv_node = conv_parallelism[index]
    
    scale_factor = numpy_helper.to_array(init_info[name])
    print(f"Scale factor: {scale_factor}")
    signed = True
    narrow = False
    bits = 8
    quant_node = {
        "name": name,
        "scale_factor": scale_factor,
        "signed": signed,
        "bits": bits,
        "narrow": narrow
    }

    pre_values = numpy_helper.to_array(init_info[name_param])
    formatted_values = parse_on_chip_weights(
        conv_node,
        quant_node,
        pre_values
    )

    tot_params -= len(formatted_values.flatten())
    print(f"Remaining params: {tot_params}")
    if concat_weights is None:
        concat_weights = pack_weights(formatted_values)
    else:
        concat_weights = np.concatenate((concat_weights, pack_weights(formatted_values)))

# Write the .npy file containting the wieghts
prj_root = "/home-ssd/roberto/Documents/nn2fpga-container/NN2FPGA/work/project_addernet_valentina/"
if concat_weights is not None:
    os.system(f"mkdir -p {prj_root}/npy/")
    np.save(f"{prj_root}/npy/addernet_weights.npy", concat_weights)
    
    concat_weights_uint8 = concat_weights.astype(np.uint8)
    # concat_weights_uint8.tofile(f"{prj_root}/npy/{file_name}_weights.bin")
    with open(f"{prj_root}/npy/addernet_weights.bin", "wb") as file:
        file.write(concat_weights_uint8.tobytes())


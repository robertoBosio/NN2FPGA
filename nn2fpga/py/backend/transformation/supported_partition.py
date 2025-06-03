import onnx
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.create_generic_partitions import PartitionFromDict

FPGA_SUPPORTED_ACTIVATIONS = {
    "Relu",
    "Sigmoid",
    "Swish"
}

FPGA_SUPPORTED_OPS = {
    "Conv",
    "Gemm",
    "Pool",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Concat",
    "Upsample",
    "Resize",
    "Add",
    "Quant",
    "IntQuant",
}

FPGA_SUPPORTED_OPS.update(FPGA_SUPPORTED_ACTIVATIONS)

def is_fpga_supported_op(model: ModelWrapper, node: onnx.NodeProto) -> bool:
    """ Check if the operation is supported by FPGA. """
    if node.op_type in FPGA_SUPPORTED_OPS:
        return True
    return False

def is_constant_input_node(model: ModelWrapper, node: onnx.NodeProto) -> bool:
    """ Check if the node is a constant input node. """
    init_names = {init.name for init in model.graph.initializer}
    return all(inp in init_names for inp in node.input)

class SupportedPartition(Transformation):
    """ Partition the model into a single partition containing only operations supported by FPGA. 
    All other operations are removed from the model. """

    def __init__(self, partition_directory: str = "partitions"):
        super().__init__()
        self.partition_directory = partition_directory

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        node_to_partition = {}
        value_to_partition = {}

        # Initial inputs and weights are assigned to FPGA
        for inp in graph.input:
            value_to_partition[inp.name] = "FPGA"

        for init in graph.initializer:
            value_to_partition[init.name] = "FPGA"

        for node in graph.node:
            partition = "CPU"

            if is_fpga_supported_op(model, node):
                # Check if all inputs are already from FPGA-allocated values
                if all(value_to_partition.get(inp, "CPU") == "FPGA" for inp in node.input):
                    partition = "FPGA"

            node_to_partition[node.name] = partition

            # Record partition for this node
            node_to_partition[node.name] = partition

            # All of this node’s outputs now inherit the same label
            for out in node.output:
                value_to_partition[out] = partition

        # Reassign constants to match their consumers
        for node in [n for n in graph.node if is_constant_input_node(model, n)]:
            op_node = model.find_consumer(node.output[0])
            if op_node is not None:
                node_to_partition[node.name] = node_to_partition[op_node.name]

        
        # Create a partition dictionary
        node_list = [node.name for node in graph.node]
        partition_dict = {
            "FPGA": [node_list.index(node) for node, part in node_to_partition.items() if part == "FPGA"]
        }

        # Create a partition from the dictionary
        model = model.transform(PartitionFromDict(partition_dict, self.partition_directory))

        return (model, False)

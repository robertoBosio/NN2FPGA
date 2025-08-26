from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.board_util import read_board_info
from onnx import helper
from qonnx.util.basic import get_by_name

class InsertAXIConverters(Transformation):
    """
    Inserts AXI converters for each input/output tensors in the model.
    This will convert the input/output tensor from/to AXI format.
    """
    def __init__(self, nn2fpga_root: str):
        """
        Initialize the transformation with the path to the nn2fpga root directory.
        
        Args:
            nn2fpga_root (str): Path to the nn2fpga root directory.
        """
        super().__init__()
        self.nn2fpga_root = nn2fpga_root
        

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        
        board_res = read_board_info(
            board=model.get_metadata_prop("board_name"),
        )

        new_nodes = []
        for i, inp in enumerate(model.graph.input):
            orig_input_name = inp.name
            produce_stream_output = f"{orig_input_name}_streamed"

            # Create the custom node NHWCToStream
            produce_node = helper.make_node(
                op_type="NHWCToStream",
                domain="backend.custom_op",
                inputs=[orig_input_name],
                outputs=[produce_stream_output],
                normalize=0,
                axi_bitwidth=board_res["axi_bitwidth"],
                name=f"NHWCToStream_{i}",
            )

            # Replace all uses of this input
            for node in model.graph.node:
                node_inputs = list(node.input)
                for j, node_in in enumerate(node_inputs):
                    if node_in == orig_input_name:
                        node.input[j] = produce_stream_output

            new_nodes.append(produce_node)

        # Insert all new nodes at the beginning
        for node in reversed(new_nodes):
            model.graph.node.insert(0, node)

        
        new_nodes = []
        for i, out in enumerate(model.graph.output):
            orig_output_name = out.name
            consume_stream_output = f"{orig_output_name}_streamed"

            # Create the custom node StreamToNHWC
            consume_node = helper.make_node(
                op_type="StreamToNHWC",
                domain="backend.custom_op",
                outputs=[consume_stream_output],
                inputs=[orig_output_name],
                axi_bitwidth=board_res["axi_bitwidth"],
                name=f"ConsumeStream_{i}"
            )

            get_by_name(model.graph.output, orig_output_name).name = consume_stream_output 
            new_nodes.append(consume_node)

        # Insert all new nodes at the beginning
        for node in new_nodes:
            model.graph.node.append(node)


        return (model, False)

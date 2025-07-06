from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.board_util import read_board_info
from onnx import helper

class InsertProduceStream(Transformation):
    """
    Inserts a ProduceStream node for each input tensor in the model.
    This node will stream the input tensor read from an AXI Lite interface.
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
            prj_root=self.nn2fpga_root
        )

        new_nodes = []
        for i, inp in enumerate(model.graph.input):
            orig_input_name = inp.name
            produce_stream_output = f"{orig_input_name}_streamed"

            # Create the custom node ProduceStream
            produce_node = helper.make_node(
                op_type="ProduceStream",
                domain="backend.custom_op",
                inputs=[orig_input_name],
                outputs=[produce_stream_output],
                normalize=0,
                axi_bitwidth=board_res["axi_bitwidth"],
                name=f"ProduceStream_{i}",
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

        return (model, False)

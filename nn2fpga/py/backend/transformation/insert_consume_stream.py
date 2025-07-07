from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.board_util import read_board_info
from onnx import helper
from qonnx.util.basic import get_by_name

class InsertConsumeStream(Transformation):
    """
    Inserts a ConsumeStream node for each output tensor in the model.
    This node will transform the streaming tensor into an AXI Lite compatible one.
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
        for i, out in enumerate(model.graph.output):
            orig_output_name = out.name
            consume_stream_output = f"{orig_output_name}_streamed"

            # Create the custom node ConsumeStream
            consume_node = helper.make_node(
                op_type="ConsumeStream",
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
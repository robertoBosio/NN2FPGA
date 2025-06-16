from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from onnx import NodeProto, helper
from collections import deque
from backend.util.par_utils import get_par_attributes, set_par_attributes, check_par_attributes
import backend.transformation as transformation


class AdjustStreamingCommunication(Transformation):
    """
    A transformation pass that adjusts the streaming communication
    between nodes by inserting necessary communication nodes adjusting
    the number of streaming channels or the channel packed in a single packet.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        # Traversing the graph to find nodes that require adjustment
        communication_nodes = []
        queue = deque(model.get_nodes_by_op_type("ProduceStream"))
        mark_visited = set()

        while queue:
            node = queue.popleft()
            mark_visited.add(node.name)
            par = get_par_attributes(node)

            # ich_par does not represent a communication parameter, so we can ignore it.
            par.pop("ich_par", None)

            consumers = model.find_direct_successors(node)
            if consumers is not None:
                for consumer in consumers:
                    if check_par_attributes(consumer):
                        consumer_par = get_par_attributes(consumer)

                        # Need to balance node.och_par -> consumer.ich_par if ich_par is present,
                        # otherwise node.och_par -> consumer.och_par.
                        # This represents the number of channels packed in a single packet.
                        if consumer_par["ich_par"] is not None:
                            consumer_ch_par = consumer_par["ich_par"]
                        else:
                            consumer_ch_par = consumer_par["och_par"]

                        if (
                            par["och_par"] != consumer_ch_par
                            or par["w_par"] != consumer_par["w_par"]
                        ):
                            # Insert a communication node.
                            communication_nodes.append(
                                (
                                    node,
                                    consumer,
                                    par["och_par"],
                                    consumer_ch_par,
                                    par["w_par"],
                                    consumer_par["w_par"],
                                )
                            )

                    if consumer.name not in mark_visited:
                        # If the consumer is not already visited, add it to the queue.
                        queue.append(consumer)
        
        model.save("pre_adjust_streaming_communication.onnx")

        for (
            producer,
            consumer,
            in_ch_par,
            out_ch_par,
            in_w_par,
            out_w_par,
        ) in communication_nodes:
            for output in producer.output:
                probable_consumer = model.find_consumer(output)
                if probable_consumer is not None and probable_consumer.name == consumer.name:
                    # Create a communication producer to adjust the streaming parameters.
                    comm_node = helper.make_node(
                        "BandwidthAdjust",
                        inputs=[output],
                        outputs=[f"{output}_comm"],
                        name=f"BandwidthAdjust_{producer.name}_{consumer.name}",
                        domain="backend.custom_op",
                        in_ch_par=in_ch_par,
                        out_ch_par=out_ch_par,
                        in_w_par=in_w_par,
                        out_w_par=out_w_par,
                    )

                    model.graph.node.append(comm_node)

                    for i, input in enumerate(consumer.input):
                        if input == output:
                            # Replace the consumer's input with the output of the communication node.
                            consumer.input[i] = f"{output}_comm"

        if communication_nodes:
            # If any communication nodes were added, we need to sort the graph
            # and infer shapes again to ensure the model is valid.
            model = model.transform(SortGraph())
            model = model.transform(transformation.CustomInferShapes())
            print(
                f"Inserted {len(communication_nodes)} communication nodes to adjust streaming communication."
            )
        return (model, False)

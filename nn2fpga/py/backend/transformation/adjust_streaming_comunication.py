from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from onnx import NodeProto
from collections import deque
from backend.util.par_utils import get_par_attributes, set_par_attributes, check_par_attributes

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

                        if par["och_par"] != consumer_ch_par or par["w_par"] != consumer_par["w_par"]:
                            # Insert a communication node.
                            communication_nodes.append(
                                (node, consumer, par["och_par"], consumer_ch_par, par["w_par"], consumer_par["w_par"])
                            )

                    if consumer.name not in mark_visited:
                    # If the consumer is not already visited, add it to the queue.
                        queue.append(consumer)
                
        for node, consumer, och_par, ich_par, w_par, consumer_w_par in communication_nodes:
            print(f"Inserting communication node between {node.name} and {consumer.name} with parameters: "
                  f"och_par={och_par}, ich_par={ich_par}, w_par={w_par}, consumer_w_par={consumer_w_par}")

        return (model, False)
            
        
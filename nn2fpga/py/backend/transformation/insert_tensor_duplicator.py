from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.custom_op.producestream import ProduceStream
from onnx import helper

class InsertTensorDuplicator(Transformation):
    """
    Inserts a TensorDuplicator node in each fork node of the model.
    This node will duplicate the tensor to ensure that each consumer gets a separate copy.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        fork_nodes = [node for node in model.graph.node if model.is_fork_node(node)]
        for node in fork_nodes:
            print(f"Inserting TensorDuplicator for fork node: {node.name}")
            
            fork_out = node.output[0]  
            consumers = model.find_consumers(fork_out)
            num_copies = len(consumers)
            dup_outputs = [f"{fork_out}_copy_{i}" for i in range(num_copies)]

            # Create the Duplicate node
            dup_node = helper.make_node(
                op_type="TensorDuplicator",
                domain="backend.custom_op",
                inputs=[fork_out],
                outputs=dup_outputs,
                name=f"Duplicate_{fork_out}",
                copies=num_copies
            )
            model.graph.node.insert(0, dup_node)  # Insert early


            # Rewire each consumer to use its own copy
            for i, consumer in enumerate(consumers):
                for j, inp_name in enumerate(consumer.input):
                    if inp_name == fork_out:
                        consumer.input[j] = dup_outputs[i]
                        modified = True

            # Copy shape and datatype info
            shape = model.get_tensor_shape(fork_out)
            dtype = model.get_tensor_datatype(fork_out)
            for out in dup_outputs:
                model.set_tensor_shape(out, shape)
                model.set_tensor_datatype(out, dtype)

        return (model, False)
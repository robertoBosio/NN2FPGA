from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from onnx import numpy_helper, helper
from qonnx.util.basic import get_by_name
from backend.util.quant_utils import get_quant_params
import numpy as np

class FoldAsymmetricActQuant(Transformation):
    """ Fold the zero point of asymmetric activation quantization into the bias of the next Conv layer. """
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        
        # Find all Conv layers in the model
        convs = model.get_nodes_by_op_type("Conv")
        for conv in convs:
            
            act_quant_params = model.get_tensor_datatype(conv.input[0])
            if act_quant_params is None or act_quant_params.zeropt == 0:
                continue  # Nothing to fold if zeropt is None or 0

            zeropt_act = act_quant_params.zeropt

            # Get the weight quantization parameters
            weight_quant = model.find_producer(conv.input[1])
            weight_quant_params = get_quant_params(weight_quant, model)
            weight_data = numpy_helper.to_array(get_by_name(graph.initializer, weight_quant.input[0]))
            
            if weight_data is None:
                print(f"Skipping {conv.name} weight data is not available.")
                continue
            
            # Sum the weights of each filter to compute the bias adjustment
            weight_sums = weight_data.reshape(weight_data.shape[0], -1).sum(axis=1)
            
            if len(conv.input) > 2:
                bias_quant = model.find_producer(conv.input[2])
                bias_quant_params = get_quant_params(bias_quant, model)

                # Reshape the weight scales to remove any extra dimensions
                weight_scales = np.squeeze(weight_quant_params["scale"])

                # Check that the bias scale is close to the product of the weight scale and activation scale
                if not all(np.isclose(bias_quant_params["scale"], 
                                  weight_scales * act_quant_params.scale)):
                    print(f"Skipping {conv.name} as bias scale does not match weight and activation scales.")
                    continue

                bias_data = numpy_helper.to_array(get_by_name(graph.initializer, bias_quant.input[0]))
                if bias_data is None:
                    print(f"Skipping {conv.name} bias data is not available.")
                    continue
            else:
                # bias_data = np.zeros(weight_data.shape[0], dtype=np.float32)
                # bias_quant = helper.make_node(
                #         "Quant",
                #         name = f"{conv.name}_bias",
                #         inputs=[out, quant_node.input[1], quant_node.input[2], quant_node.input[3]],  # Use the same quantization parameters
                #         outputs=[new_quant_output],
                #         domain=quant_node.domain,
                #     )
                print(f"Skipping {conv.name} as it has no bias input.")
                continue

            # Adjust the bias by subtracting the zero point scaled by the weight sums
            new_bias_data = bias_data - (zeropt_act * act_quant_params.scale * weight_sums)
            model.set_initializer(bias_quant.input[0], new_bias_data)

            # Remove the zero point input from the activation quantization node
            conv.attribute.append(
                helper.make_attribute(key="asym_folded", value=1, attr_type=helper.AttributeProto.INT)
            )

            print(f"Folded zero point of asymmetric activation quantization into bias of {conv.name}.")
        return (model, False)



                 
    
    
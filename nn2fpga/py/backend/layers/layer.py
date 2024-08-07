from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Layer class represents a layer in the neural network

    Attributes:
        name (str): Name of the layer
        type (str): Type of the layer
        width_par (int): Width parallelism of the layer
        height_par (int): Height parallelism of the layer
        channel_par (int): Channel parallelism of the layer
        foldable (bool): Indicates if the layer can be folded

    Note:

    """

    def __init__(self, name):
        self.name = name
        self.type = None
        self.width_par = 1
        self.height_par = 1
        self.channel_par = 1
        self.foldable = False

    @abstractmethod
    def parse(self):
        pass

class Net:
    """ 
    Net class represents a communication stream between layers 

    Attributes:
        name (str): Name of the net
        depth (int): Depth of the hls::stream
        width (int): Array dimension of the hls::stream
        channel_packing (int): Number of channels packed in a single element of the hls::stream
        data_type (QuantLayer): Data type of the hls::stream
        N (int): Number of elements in the first dimension of the tensor
        C (int): Number of elements in the second dimension of the tensor
        H (int): Number of elements in the third dimension of the tensor
        W (int): Number of elements in the fourth dimension of the tensor
   
    Note:
    
    """

    def __init__(self, name, onnx_tensor_shape):
        self.name = name
        self.depth = 1  # Depth of the hls::stream
        self.width = 1  # Array dimension of the hls::stream
        self.channel_packing = 1  # Number of channels packed in a single element of the hls::stream
        self.data_type = None # Data type of the hls::stream

        self.N = 1
        self.C = 1
        self.H = 1
        self.W = 1
        if len(getattr(onnx_tensor_shape, 'dim')) > 0:
            self.N = getattr(onnx_tensor_shape, 'dim')[0].dim_value
        if len(getattr(onnx_tensor_shape, 'dim')) > 1:
            self.C = getattr(onnx_tensor_shape, 'dim')[1].dim_value
        if len(getattr(onnx_tensor_shape, 'dim')) > 2:
            self.H = getattr(onnx_tensor_shape, 'dim')[2].dim_value
        if len(getattr(onnx_tensor_shape, 'dim')) > 3:
            self.W = getattr(onnx_tensor_shape, 'dim')[3].dim_value
        if len(getattr(onnx_tensor_shape, 'dim')) > 4:
            raise ValueError(f"Unknown shape with more than 4 dimensions.")

    def __str__(self) -> str:
        return f"Net: {self.name}, \
                \n\tTensor shape: ({self.N}, {self.C}, {self.H}, {self.W}), \
                \n\tDepth: {self.depth}, \
                \n\tWidth: {self.width}, \
                \n\tChannel packing: {self.channel_packing}, \
                \n\tData type: {self.data_type}"

    def parse(self):
        pass
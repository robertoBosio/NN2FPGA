from abc import ABC, abstractmethod

class Layer(ABC):

    def __init__(self, name, input_shape, output_shape):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input = []
        self.output = []
        self.type = None

    @abstractmethod
    def parse(self):
        pass
from .layer import Layer
import numpy as np
import mlx.core as mx

class FlattenLayer(Layer):
    def __init__(self):
        pass

    def print(self):
        print("Flatten Layer")

    def forward_prop(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward_prop(self, output_error, learning_rate):
        return output_error.reshape(self.input_shape)
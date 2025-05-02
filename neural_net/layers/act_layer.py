from .layer import Layer
import numpy as np
import mlx.core as mx
from typing import Literal

class ActivationLayer(Layer):
    def __init__(self, activation: Literal['tanh', 'sigmoid', 'relu', 'binary_step']):
        act, act_prime = self.get_activations(activation)
        self.activation =  act
        self.activation_prime = act_prime

    def forward_prop(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward_prop(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error # dE/dX_i=dE/dY_i*dY_i/dX_i=dE/dY_i*f'(X_i) -> dE/dX=dE/dY elem* f'(X)
    
    def print(self):
        print("Activation Layer")

    def get_activations(self, activation):
        if activation == 'tanh':
            def tanh(x):
                return mx.tanh(x)
            def tanh_prime(x):
                return 1-mx.tanh(x)**2
            return tanh, tanh_prime
        elif activation == 'sigmoid':
            def sigmoid(x):
                return 1/(1+mx.exp(-x))
            def sigmoid_prime(x):
                return sigmoid(x)*(1-sigmoid(x))
            return sigmoid, sigmoid_prime
        elif activation == 'relu':
            def relu(x):
                return mx.maximum(0, x)
            def relu_prime(x):
                return mx.where(x>0, 1, 0)
            return relu, relu_prime
        elif activation == 'binary_step':
            def binary_step(x):
                return mx.where(x>=0, 1, 0)
            def binary_step_prime(x):
                return 0
            return binary_step, binary_step_prime
        
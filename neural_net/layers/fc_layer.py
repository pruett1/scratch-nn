from .layer import Layer
import numpy as np
import mlx.core as mx

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        #use kaiming initialization
        limit = 2 / input_size
        self.weights = mx.random.normal(scale=limit, shape=[input_size, output_size])
        self.bias = mx.zeros(shape=[1, output_size])

    def print(self):
        print("fc layer")

    def forward_prop(self, input):
        self.input = input
        self.output = mx.matmul(self.input, self.weights) + self.bias
        return self.output
    
    # calculates deriv error wrt to params, updates params and returns deriv error wrt input
    def backward_prop(self, output_error, learning_rate):
        input_error = mx.matmul(output_error, self.weights.T) # dE/dX_i=sum{n=1,j}(dE/dY_n*dY_n/dX_i)=sum{n=1,j}(dE/dY_n*W_in)=dE/dY*W^T
        weights_error = mx.matmul(self.input.T, output_error) # dE/dW_ij=sum{n=1,j}(dE/dY_n*dY_n/dW_ij)=dE/dY_j*X_i -> dE/dW = X_i^T * dE/dY

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * mx.sum(output_error, keepdims=True) # dE/dB_j=sum{n=1,j}(dE/dY_n*dY_n/dB_j)=dE/dY_j -> dE/dB = dE/dY
        return input_error
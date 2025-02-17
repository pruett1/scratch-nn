from .layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_prop(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    # calculates deriv error wrt to params, updates params and returns deriv error wrt input
    def backward_prop(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T) # dE/dX_i=sum{n=1,j}(dE/dY_n*dY_n/dX_i)=sum{n=1,j}(dE/dY_n*W_in)=dE/dY*W^T
        weights_error = np.dot(self.input.T, output_error) # dE/dW_ij=sum{n=1,j}(dE/dY_n*dY_n/dW_ij)=dE/dY_j*X_i -> dE/dW = X_i^T * dE/dY

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error # dE/dB_j=sum{n=1,j}(dE/dY_n*dY_n/dB_j)=dE/dY_j -> dE/dB = dE/dY
        return input_error
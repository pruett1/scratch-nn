from .layer import Layer
import numpy as np
import mlx.core as mx

class FCLayer(Layer):
    def __init__(self, input_size, output_size, optimizer='adam', beta1=0.9, beta2=0.999, epsilon=1e-8):
        #use kaiming initialization
        limit = 2 / input_size
        self.weights = mx.random.normal(scale=limit, shape=[input_size, output_size])
        self.bias = mx.zeros(shape=[1, output_size])

        self.optimizer = optimizer

        if optimizer == 'adam':
            self.m_w = mx.zeros_like(self.weights)
            self.m_b = mx.zeros_like(self.bias)
            self.v_w = mx.zeros_like(self.weights)
            self.v_b = mx.zeros_like(self.bias)
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.t = 0

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
        bias_error = mx.sum(output_error, axis=0, keepdims=True) # dE/dB_j=sum{n=1,j}(dE/dY_n*dY_n/dB_j)=dE/dY_j -> dE/dB = dE/dY

        # update parameters
        if self.optimizer == 'sgd':
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * bias_error
        elif self.optimizer == 'adam':
            self.t += 1
            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_error
            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * mx.square(weights_error)
            m_hat_w = self.m_w / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w / (1 - self.beta2 ** self.t)
            self.weights -= learning_rate * m_hat_w / (mx.sqrt(v_hat_w) + self.epsilon)
            
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * bias_error
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * mx.square(bias_error)
            m_hat_b = self.m_b / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b / (1 - self.beta2 ** self.t)
            self.bias -= learning_rate * m_hat_b / (mx.sqrt(v_hat_b) + self.epsilon)

        return input_error
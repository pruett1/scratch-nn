from .layer import Layer
import numpy as np
import mlx.core as mx

class MaxPoolLayer2d(Layer):
    def __init__(self, kernel_size, stride=1):
        self.kernel_size = kernel_size
        self.stride = stride

    def print(self):
        print("Max Pooling Layer")

    def forward_prop(self, input):
        self.input = input
        batch_size, channels, height, width = input.shape
        out_h = (height - self.kernel_size) // self.stride + 1
        out_w = (width - self.kernel_size) // self.stride + 1

        out = mx.zeros(shape=[batch_size, channels, out_h, out_w])
        input_exp = mx.zeros(shape=[batch_size, channels, self.kernel_size, self.kernel_size, out_h, out_w])

        #temp 6d to apply max on
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                input_exp[:, :, i, j, :, :] = input[:, :, i:i+self.stride*out_h:self.stride, j:j+self.stride*out_w:self.stride]

        out = mx.max(input_exp, axis=(2, 3)) #max reduction (shape [batch_size, channels, out_h, out_w])
        
        # Store the indices of the max values for backpropagation
        self.max_indices = (input_exp == mx.max(input_exp, axis=(2, 3), keepdims=True)) 

        return out
    
    def backward_prop(self, output_error, learning_rate):
        batch_size, channels, out_h, out_w = output_error.shape
        input_error = mx.zeros_like(self.input)

        output_error = output_error[:,:,None,None,:,:] #expand output error to match one-hot max indices mask
        output_error = mx.multiply(output_error, self.max_indices) #multiply by the max indices

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                input_error[:, :, i:i+self.stride*out_h:self.stride, j:j+self.stride*out_w:self.stride] += output_error[:,:,i,j,:,:]

        return input_error
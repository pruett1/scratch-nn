from .layer import Layer
import numpy as np
import mlx.core as mx

#simple convolutional layer with a stride of 1
class Conv2dLayer(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, optimizer='adam', beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.padding = padding
        self.out_channels = out_channels
        if (bias):
            self.bias = mx.zeros(shape=[1, out_channels])
        else:
            self.bias = None

        #use Kaiming initialization
        limit = 2 / (in_channels * kernel_size * kernel_size)
        self.kernel = mx.random.normal(scale=limit, shape=[out_channels, in_channels, kernel_size, kernel_size])

        self.optimizer = optimizer

        if optimizer == 'adam':
            self.m_k = mx.zeros_like(self.kernel)
            self.m_b = mx.zeros_like(self.bias)
            self.v_k = mx.zeros_like(self.kernel)
            self.v_b = mx.zeros_like(self.bias)
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.t = 0

    def print(self):
        print("conv layer")

    def forward_prop(self, input):
        # Assuming input is of shape (batch_size, in_channels, height, width)
        # and kernel is of shape (out_channels, in_channels, kernel_size, kernel_size)
        # Implemented using Image2Col and matrix multiplication

        # Pad by amount specified in self.padding
        if (self.padding > 0):
                input = mx.pad(input, [(0, 0), (0,0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')

        self.input = input
        self.batch_size = input.shape[0]
        input_height, input_width = input.shape[2], input.shape[3]
        self.out_shape = (self.out_channels, input_height - self.kernel.shape[2] + 1, input_width - self.kernel.shape[3] + 1)

        # Perform the convolution operation
        self.col = self.image2col(input)

        self.kernel_vec = self.kernel.reshape(self.out_channels, -1).T

        out =  mx.matmul(self.col, self.kernel_vec) #shape (batch_size * out_h * out_w, out_channels)

        if (self.bias is not None):
            out += self.bias

        out = out.reshape(self.batch_size, self.out_shape[1], self.out_shape[2], self.out_channels).transpose(0,3, 1, 2)

        return out
    
    def backward_prop(self, output_error, learning_rate):
        dout = output_error.transpose(0, 2, 3, 1).reshape(self.batch_size * self.out_shape[1] * self.out_shape[2], self.out_channels)

        #compute gradients
        self.db = mx.sum(dout, axis=0, keepdims=True) #gradient wrt bias
        self.dkernel = mx.matmul(self.col.T, dout)
        self.dkernel = self.dkernel.transpose(1, 0).reshape(self.out_channels, self.input.shape[1], self.kernel.shape[2], self.kernel.shape[3]) #gradient wrt kernel

        dcol = mx.matmul(dout, self.kernel_vec.T) #gradient wrt col

        if self.optimizer == 'sgd':
            #gradient descents
            if (self.bias is not None):
                self.bias -= learning_rate * self.db

            self.kernel -= learning_rate * self.dkernel
        elif self.optimizer == 'adam':
            self.t += 1
            self.m_k = self.beta1 * self.m_k + (1 - self.beta1) * self.dkernel
            self.v_k = self.beta2 * self.v_k + (1 - self.beta2) * mx.square(self.dkernel)
            m_hat_k = self.m_k / (1 - self.beta1 ** self.t)
            v_hat_k = self.v_k / (1 - self.beta2 ** self.t)
            self.kernel -= learning_rate * m_hat_k / (mx.sqrt(v_hat_k) + self.epsilon)
            
            if self.bias is not None:
                self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * self.db
                self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * mx.square(self.db)
                m_hat_b = self.m_b / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_b / (1 - self.beta2 ** self.t)
                self.bias -= learning_rate * m_hat_b / (mx.sqrt(v_hat_b) + self.epsilon)

        dx = self.col2image(dcol, self.padding) #gradient wrt input

        return dx
    
    
    def image2col(self, input):
        out_h = self.out_shape[1]
        out_w = self.out_shape[2]
        #Initialize the column matrix
        col = mx.zeros((self.batch_size, self.input.shape[1], self.kernel.shape[2], self.kernel.shape[3], out_h, out_w))

        #Extract the patches from the input covered by the kernel
        for y in range(self.kernel.shape[2]):
            for x in range(self.kernel.shape[3]):
                col[:,:,y,x,:,:] = input[:,:,y:y+out_h,x:x+out_w]

        #Reshape col from 6d to 2d
        #Each row is the flattened image patch C*F_H*F_W and each row will be multiplied with the flattened kernel vec
        return col.transpose(0, 4, 5, 1, 2, 3).reshape(self.batch_size * out_h * out_w, -1)

    def col2image(self, col, padding):
        out_h = self.out_shape[1]
        out_w = self.out_shape[2]
        H = self.input.shape[2] - padding*2
        W = self.input.shape[3] - padding*2

        col = col.reshape(self.batch_size, self.input.shape[1], self.kernel.shape[2], self.kernel.shape[3], out_h, out_w)

        img = mx.zeros((self.input.shape)) #stride of 1 so img will be same size as input (and padding is already done)

        for y in range(self.kernel.shape[2]):
            for x in range(self.kernel.shape[3]):
                img[:,:,y:y+out_h,x:x+out_w] += col[:,:,y,x,:,:]

        return img[:,:,padding:H+padding,padding:W+padding] #remove padding
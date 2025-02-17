import numpy as np
from typing import Literal

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)
    
    # set loss
    def loss_type(self, loss_name: Literal['mse']):
        loss, loss_prime = self.get_loss(loss_name)
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        # sample dimension
        samples = len(input)
        result = []

        # run over samples
        for i in range(samples):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_prop(output)
            result.append(output)

        return result
    
    # train network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension
        samples = len(x_train)
        
        # training loop
        for i in range(epochs):
            err =  0
            for j in range(samples):
                # forward prop
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)
                
                # compute loss
                err += self.loss(y_train[j], output)

                # backward prop
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, learning_rate)

            err /= samples
            print('epoch %d/%d  error=%f' % (i, epochs, err))

    def get_loss(self, loss_name):
        if loss_name == 'mse':
            def mse(y, y_hat):
                return np.mean(np.power(y-y_hat, 2))

            def mse_prime(y, y_hat):
                return 2*(y_hat-y)/y.size
            
            return mse, mse_prime
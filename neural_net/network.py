import numpy as np
import mlx.core as mx
from typing import Literal
import time
import matplotlib.pyplot as plt

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

    def predict(self, input, batch_size=64):
        # sample dimension
        samples = len(input)
        result = []

        # run over samples
        for i in range(0, samples, batch_size):
            batch_end = min(i+batch_size, samples)
            input_batch = input[i:batch_end]
            output = input_batch
            for layer in self.layers:
                output = layer.forward_prop(output)
            result.append(output)

        # Concatenate results using mlx.core
        return mx.concatenate(result, axis=0)
    
    # train network
    def fit(self, x_train, y_train, epochs, learning_rate, decay_factor, batch_size=64, plot_loss=False):
        if (plot_loss):
            plt.ion()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], label='loss')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('Training Loss')
            ax.legend()
            plt.show()

            epoch_list = []
            loss_list = []
        
        # sample dimension
        samples = len(x_train)
        steps_per_epoch = samples // batch_size
        
        # training loop
        for i in range(epochs):
            err =  0

            for j in range(0, samples, batch_size):
                batch_end = min(j+batch_size, samples)
                # Get batch
                x_batch = x_train[j:batch_end]
                y_batch = y_train[j:batch_end]

                # Forward pass
                output = x_batch
                for layer in self.layers:
                    output = layer.forward_prop(output)

                # Compute loss and accumulate error
                err += self.loss(y_batch, output)

                # Backward pass
                error = self.loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, learning_rate)

            err /= steps_per_epoch
            learning_rate = learning_rate / (1 + decay_factor * i)
            print('epoch %d/%d  error=%f' % (i, epochs, err))

            if (plot_loss):
                loss_list.append(err)
                epoch_list.append(i)
                line.set_xdata(np.array(mx.array(epoch_list)))
                line.set_ydata(np.array(mx.array(loss_list)))
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)
        
        if (plot_loss):
            plt.ioff()
            plt.show()

    def get_loss(self, loss_name):
        if loss_name == 'mse':
            def mse(y, y_hat):
                # print(y-y_hat)
                # print(mx.power(y-y_hat, 2))
                # print(mx.mean(mx.power(y-y_hat, 2)))
                return mx.mean(mx.power(y-y_hat, 2))

            def mse_prime(y, y_hat):
                return 2*(y_hat-y)/y.size
            
            return mse, mse_prime
        elif (loss_name == 'cross_entropy'):
            def cross_entropy(y, y_hat):
                # y: (batch_size, ) integer labels
                # y_hat: (batch_size, num_classes) softmax probabilities
                batch_indices = mx.arange(y.shape[0])
                correct_class_probs = y_hat[batch_indices, y]
                loss = -mx.log(correct_class_probs + 1e-9) #add small value to avoid log(0)
                return mx.mean(loss)
            
            def cross_entropy_prime(y, y_hat):
                # y: (batch_size, ) integer labels
                # y_hat: (batch_size, num_classes) softmax probabilities

                y_one_hot = mx.zeros_like(y_hat)
                y_one_hot[mx.arange(y.shape[0]), y] = 1
                grad = y_hat - y_one_hot
                return grad / y.shape[0] # normalize by batch size
            
            return cross_entropy, cross_entropy_prime
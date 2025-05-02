import numpy as np
import mlx.core as mx

from neural_net.network import Network
from neural_net.layers.fc_layer import FCLayer
from neural_net.layers.act_layer import ActivationLayer

# training data
x_train = mx.array([[0,0], [0,1], [1,0], [1,1]])
y_train = mx.array([[0], [1], [1], [0]])

net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer('tanh'))
net.add(FCLayer(3, 12))
net.add(ActivationLayer('tanh'))
net.add(FCLayer(12, 24))
net.add(ActivationLayer('tanh'))
net.add(FCLayer(24, 12))
net.add(ActivationLayer('relu'))
net.add(FCLayer(12, 3))
net.add(ActivationLayer('tanh'))
net.add(FCLayer(3, 1))
net.add(ActivationLayer('sigmoid'))

net.loss_type('mse')
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

out = net.predict(x_train)
print(out)


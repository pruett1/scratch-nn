import numpy as np
import mlx.core as mx
from sklearn.metrics import classification_report, accuracy_score

from neural_net.network import Network
from neural_net.layers.fc_layer import FCLayer
from neural_net.layers.act_layer import ActivationLayer
from neural_net.layers.con_layer import Conv2dLayer
from neural_net.layers.max_pool import MaxPoolLayer2d
from neural_net.layers.flatten import FlattenLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Convert to mlx array
x_train = mx.array(x_train) / 255 #normalize
x_test = mx.array(x_test) / 255 #normalize
y_train = mx.array(y_train)
y_test = mx.array(y_test)

# # Reshape y_train one-hot encoding
# num_classes = 10
# one_hot = mx.zeros((y_train.shape[0], num_classes))
# one_hot[mx.arange(y_train.shape[0]), y_train] = 1
# y_train = one_hot

#Reshape x_train and x_test to be shape (batch_size, channels, height, width)
if x_train.ndim == 2:
    x_train = x_train[np.newaxis, np.newaxis, :, :]  # → (1, 1, H, W)
elif x_train.ndim == 3:
    x_train = x_train[:, np.newaxis, :, :]           # → (1, C, H, W)

if x_test.ndim == 2:
    x_test = x_test[np.newaxis, np.newaxis, :, :]  # → (1, 1, H, W)
elif x_test.ndim == 3:
    x_test = x_test[:, np.newaxis, :, :]           # → (1, C, H, W)

net = Network()
net.add(Conv2dLayer(1, 32, 3, padding=1))
net.add(ActivationLayer('relu'))
net.add(Conv2dLayer(32, 64, 3, padding=1))
net.add(ActivationLayer('relu'))
net.add(MaxPoolLayer2d(2, 2))
net.add(FlattenLayer())
net.add(FCLayer(64*14*14, 128))
net.add(ActivationLayer('relu'))
net.add(FCLayer(128, 10))
net.add(ActivationLayer('softmax'))

net.loss_type('cross_entropy')
net.fit(x_train, y_train, epochs=450, learning_rate=0.001, decay_factor=0.001, batch_size=128, plot_loss=True)

out = net.predict(x_test)
y_pred = mx.argmax(out, axis=1).tolist()
y_true = y_test.tolist()


classification_report = classification_report(y_pred, y_true)
print(classification_report)
accuracy = accuracy_score(y_pred, y_true)
print(f'Accuracy: {accuracy * 100:.2f}%')

print(out)
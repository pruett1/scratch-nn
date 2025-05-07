import numpy as np
import mlx.core as mx
import time
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
start_time = time.time()

num_epochs = net.fit(x_train, y_train, epochs=60, learning_rate=0.001, decay_factor=0.01, batch_size=128, plot_loss=False, 
        early_stopping=True, early_stopping_threshold=0.01, patience=5)

end_time = time.time()
scratch_training_time = end_time - start_time
print(f'Training time: {scratch_training_time:.2f} seconds')

out = net.predict(x_test)
y_pred = mx.argmax(out, axis=1).tolist()
y_true = y_test.tolist()

class_report = classification_report(y_pred, y_true)
print(class_report)
accuracy_scratchnn = accuracy_score(y_pred, y_true)
print(f'Accuracy: {accuracy_scratchnn * 100:.2f}%')

torch_model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64*14*14, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
    # no activation layer here, as we will use CrossEntropyLoss which applies softmax internally
)

torch_model = torch_model.to('mps')

torch_model.train()
optimizer = optim.Adam(torch_model.parameters(), lr=0.001, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

rolling_error = []
patience = 10
early_stopping_threshold = 0.01
early_stopping = False

start_time = time.time()
for epoch in range(num_epochs):
    for i in range(0, len(x_train), batch_size):
        batch_end = min(i+batch_size, len(x_train))
        x_batch = torch.tensor(x_train[i:batch_end], dtype=torch.float32).to('mps')
        y_batch = torch.tensor(y_train[i:batch_end], dtype=torch.long).to('mps')

        optimizer.zero_grad()
        output = torch_model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()

    if early_stopping:
        if len(rolling_error) == patience:
            best_loss = min(rolling_error)
            improvement = (loss.item() - best_loss) / best_loss

            if improvement >= -early_stopping_threshold:
                print(f"Early stopping at epoch {epoch+1}")
                break

        rolling_error.append(loss.item())
        if len(rolling_error) > patience:
            rolling_error = rolling_error[1:]

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

end_time = time.time()
torch_training_time = end_time - start_time
print(f'Training time: {torch_training_time:.2f} seconds')
torch_model.eval()


x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to('mps')
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to('mps')
with torch.no_grad():
    output = torch_model(x_test_tensor)
    _, predicted = torch.max(output, 1)
    y_pred_torch = predicted.cpu().numpy()
    y_true_torch = y_test_tensor.cpu().numpy()
    class_report = classification_report(y_true_torch, y_pred_torch)
    print(class_report)
    accuracy_torch = accuracy_score(y_true_torch, y_pred_torch)
    print(f'Accuracy: {accuracy_torch * 100:.2f}%')

# Compare the accuracy of the two models
print(f'Accuracy of ScratchNN: {accuracy_scratchnn * 100:.2f}%')
print(f'Accuracy of PyTorch: {accuracy_torch * 100:.2f}%')
print(f'Accuracy difference: {accuracy_scratchnn - accuracy_torch * 100:.2f}%')
print(f'Accuracy % change: {(accuracy_torch - accuracy_scratchnn) / accuracy_torch:.2f}%')

print(f'Training time difference: {torch_training_time - scratch_training_time:.2f} seconds')
print(f'Training time % change: {(torch_training_time - scratch_training_time) / torch_training_time * 100:.2f}%')
# scratch-nn
This repository is an implementation of a basic neural network written in Python with the following layers available
- 2d Convolutional
- Fully Connected
- 2d Max Pool
- Flatten
- Activation with the RELU, softmax (assumes cross-entropy loss), tanh, sigmoid, and binary-step  activation functions

Additionally, the option for the optimizer is either Adapative Movement Estimation (Adam) and Simple Gradient Descent (SGD)

This neural network is compared to a PyTorch model of the same structure on the MNIST dataset

## Requirements
### Hardware
The MLX library was used to speed up matrix multiplication
- Apple Silicon Mac (M1+)

### Packages
All required packages are in requirements.txt (versions are from what was used during development)
- MLX
- Numpy
- PyTorch
- Scikit-learn
- Keras datasets
- Matplotlib

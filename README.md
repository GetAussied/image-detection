# image-detection

A Python implementation of a feedforward neural network with mini-batch training (using CPU).

This project implements a feedforward neural network with:
- Support for multiple hidden layers
- Mini-batch training for scalability
- Forward and backward propagation
- Save/load functionality for model parameters

Example:

from network import HiddenLayer, GenerateBatch

batches = GenerateBatch(category=[['cats', [1,0]], ['dogs', [0,1]]], batch_size=32)
layer = HiddenLayer(input_size=512*512, output_size=128, learning_rate=0.01)

for images, labels in batches:
    output = layer.forward(images)
    grad_input = layer.backward(output - labels)

Method:
### HiddenLayer
- `forward(input_data)`: forward pass with sigmoid
- `backward(grad_output)`: compute gradients and update parameters
- `save(path)`: save model parameters
- `load(path)`: load model parameters

### GenerateBatch
- Loads, shuffles, and generates mini-batches of data

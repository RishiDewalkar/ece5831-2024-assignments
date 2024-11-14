import numpy as np
from activations import Activations
from errors import Errors
from collections import OrderedDict
from layers import SoftmaxWithLoss, Affine, Relu

class TwoLayerNetWithBackProp:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights and biases
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Add layers and activation functions
        self.activations = Activations()
        self.errors = Errors()
        self.layers = OrderedDict()
        self.update_layers()
        
        # Last layer: softmax with cross-entropy loss
        self.last_layer = SoftmaxWithLoss()

    def update_layers(self):
        # Define layers: Affine -> ReLU -> Affine
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])

    def predict(self, x):
        # Forward pass through layers
        for layer in self.layers.values():
            x = layer.forward(x)
        return x  # Output predictions (batch_size, output_size)

    def loss(self, x, y):
        # Calculate the loss
        y_hat = self.predict(x)
        return self.last_layer.forward(y_hat, y)  # Softmax + Cross-entropy

    def accuracy(self, x, y):
        # Compute accuracy
        y_hat = self.predict(x)
        p = np.argmax(y_hat, axis=1)
        y_p = np.argmax(y, axis=1)
        return np.sum(p == y_p) / float(x.shape[0])

    def gradient(self, x, y):
        # Perform forward pass to calculate the loss
        self.loss(x, y)

        # Backward pass
        dout = 1  # Derivative of the loss with respect to itself
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Store gradients for weights and biases
        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads

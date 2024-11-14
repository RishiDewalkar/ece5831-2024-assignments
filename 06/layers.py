import numpy as np
from activations import Activations  # assuming this has a `sigmoid` method
from errors import Errors  # assuming this has a `cross_entropy_error` method

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))  # Sigmoid function directly implemented
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out  # Gradient of sigmoid
        return dx

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # Flatten input if necessary
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # Reshape dx to original input shape
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.y_hat = None
        self.y = None
        self.loss = None

    def forward(self, y_hat, y):
        # Flatten y_hat if it has more than 2 dimensions
        if y_hat.ndim > 2:
            y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[-1])

        self.y = y
        self.y_hat = y_hat
        self.loss = Errors().cross_entropy_error(self.y_hat, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.y_hat - self.y) / batch_size
        return dx

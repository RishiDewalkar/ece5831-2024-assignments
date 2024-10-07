"""
@author: Rishikesh Dewalkar
"""
import numpy as np

class MultilayerPerception:
    def __init__(self, w1, b1, w2, b2, w3, b3):
        self.net = {}

        self.net['w1'] = w1
        self.net['b1'] = b1

        self.net['w2'] = w2
        self.net['b2'] = b2

        self.net['w3'] = w3
        self.net['b3'] = b3

    def sigmoid(self, a):
        return 1/(1 + np.exp(-a))
    
    def forward(self, x):
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']
        
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)
        
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)
        
        a3 = np.dot(z2, w3) + b3

        return a3
    
if __name__ == "__main__":

    print("The result of matrix multiplication is computed by the multilayer percpetron class using a three-layer neural network algorithm.")
    x = np.array([0.5, 1])
    print("\n\nThe given array is the input to the neural network")
    print(x)

    print("The various sets of weights and biases used to determine the second layer nodes are as follows:")
    w1 = np.array([[0.1, 0.2, 0.15], [0.6, 0.2, 0.8]])
    b1 = np.array([0.1, 0.2, 0.05])
    print("Weights w1: ", w1)
    print("Bias b1: ", b1)

    print("The various sets of weights and biases used to determine the third layer nodes are as follows:")
    w2 = np.array([[0.1, 0.2], [0.05, 0.5], [0.1, 0.2]])
    b2 = np.array([0.4, 0.5])
    print("weight w2: ", w2)
    print("bias b2: ", b2)
    
    print("The following weights and bias is used to calculate the output nodes.")
    w3 = np.array([[0.1, 0.2], [0.9, 0.8]])
    b3 = np.array([0.9, 0.9])
    print("weight w3: ",w3)
    print("bias b3: ",b3)
    
    print("The multilayer perceptron class's forward function will obtain the output by applying the previously mentioned weights and biases to each input and hidden layer.")
    print("\n\nFor instance: The level 1 node contains the inputs in numpy array forms.")
    print("A variety of weights and biases, including w1, w2, w3, and b1, b2, b3, are provided as inputs at layer 0.")
    print("From those inputs we get a1 at layer 1 and then applying activation function to get z1 of that node.")
    print("The previous z1 is then used to calculate z2 of node 2 at the second layer by applying sigmoid activation function to a2.")
    print("The output is shown at the output layer using z2 and third pair of weights and biases.")
    y = MultilayerPerception(w1, b1, w2, b2, w3, b3)
    z = y.forward(x)
    print("\n\nThe output for the above multilayer neural network is z = ",z)
    
    
"""
@author: Rishikesh Dewalkar
"""
import numpy as np

from multilayer_perceptron import MultilayerPerception

w1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]) 
b1 = np.array([0.1, 0.2, 0.1])

w2 = np.array([[0.1, 0.3], [0.4, 0.6], [0.1, 0.4]])
b2 = np.array([0.1, 0.1])

w3 = np.array([[0.2, 0.3],[0.5, 0.6]])
b3 = np.array([0.1, 0.2]) #bias 3

mlp = MultilayerPerception(w1, b1, w2, b2, w3, b3) 

x = np.array([7, 9]) # Inputs
y = mlp.forward(x) # outputs

print(y)
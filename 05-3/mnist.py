import mnist_data
import numpy as np
import pickle


class Mnist:
    def __init__(self):
        self.data = mnist_data.MnistData()
        self.params = {}

    def sigmoid(self, x):
        # Sigmoid function with stability adjustments
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def softmax(self, a):
        # Softmax function with stability adjustments
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a / np.sum(exp_a)

    def load(self):
        # Load dataset and return training and testing data
        return self.data.load()

    def init_network(self):
        # Load pre-trained weights from file
        try:
            with open('model/sample_weight.pkl', 'rb') as f:
                self.params = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Error: Model weights file 'sample_weight.pkl' not found in the 'model' directory.")
        except Exception as e:
            raise Exception(f"Error loading model weights: {e}")

    def predict(self, x):
        # Perform forward pass prediction using pre-trained weights
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.softmax(a3)
        return y

    def accuracy(self, x, t):
        """Compute accuracy by comparing predictions with true labels."""
        t = np.array(t).flatten()  # Ensure true labels are in a 1D array

        accuracy_cnt = 0
        for i in range(len(x)):
            y = self.predict(x[i].reshape(1, -1))  # Reshape each input for single prediction
            p = np.argmax(y)  # Get the index of the highest probability

            # Check if the prediction matches the true label
            if p == t[i]:
                accuracy_cnt += 1

        return accuracy_cnt / len(x)  # Return accuracy as a fraction

if __name__ == '__main__':
    # Initialize Mnist object and load data
    mnist = Mnist()
    (x_train, y_train), (x_test, y_test) = mnist.load()

    mnist.init_network()
    accuracy = mnist.accuracy(x_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")  # Display accuracy as a percentage

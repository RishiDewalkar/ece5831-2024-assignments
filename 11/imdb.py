"""
@author: Rishikesh Atul Dewalkar
"""

# Import packages
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Default values
NUM_WORDS = 10000
BATCH_SIZE = 512


class Imdb:
    def __init__(self, num_words=NUM_WORDS):
        self.num_words = num_words
        self.x_train, self.y_train, self.x_test, self.y_test = self._load(num_words)
        self.model = self._model()

    def _load(self, num_words):
        # Load the IMDB dataset
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.load_data(num_words=num_words)

        # Vectorize reviews
        x_train = self._vectorize_sequences(x_train_raw, num_words)
        y_train = np.asarray(y_train_raw).astype('float32')

        x_test = self._vectorize_sequences(x_test_raw, num_words)
        y_test = np.asarray(y_test_raw).astype('float32')

        return x_train, y_train, x_test, y_test

    def _vectorize_sequences(self, sequences, dimension):
        # Create a binary matrix for one-hot encoding
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0  # Use advanced indexing for efficiency
        return results

    def _model(self):
        # Define the neural network model
        model = keras.Sequential([
            layers.Dense(32, activation="relu", input_shape=(self.num_words,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])

        # Compile the model
        model.compile(optimizer="rmsprop",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def train(self, epochs=20):
        if self.model is None:
            print('[INFO] Model is not defined.')
            return None

        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=BATCH_SIZE,
            verbose=2  # Provide progress updates
        )
        return history

    def evaluate(self):
        # Evaluate the model on the test data
        score = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print(f'[INFO] Test loss: {score[0]}')
        print(f'[INFO] Test accuracy: {score[1]}')

    def plot_loss(self, history):
        # Plot training and validation loss
        history_dict = history.history
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss, "r", label="Training Loss")
        plt.plot(epochs, val_loss, "r--", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self, history):
        # Plot training and validation accuracy
        history_dict = history.history
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, acc, "b", label="Training Accuracy")
        plt.plot(epochs, val_acc, "b--", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


# Main script to run the Imdb class
if __name__ == "__main__":
    imdb_model = Imdb()
    history = imdb_model.train(epochs=10)  # Train the model
    imdb_model.evaluate()  # Evaluate the model
    imdb_model.plot_loss(history)  # Plot training and validation loss
    imdb_model.plot_accuracy(history)  # Plot training and validation accuracy

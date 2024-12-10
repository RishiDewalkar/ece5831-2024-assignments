import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt

# default values
NUM_WORDS = 10000
BATCH_SIZE = 512

class Reuters:
    def __init__(self, num_words=NUM_WORDS):
        self.num_words = num_words
        self.x_train, self.y_train, \
        self.x_test, self.y_test = None, None, None, None
        self.model = None
        print("[INFO] Initialized Reuters class.")

    def prepare_data(self):
        print("[INFO] Preparing data...")
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = reuters.load_data(num_words=self.num_words)

        # vectorize reviews
        self.x_train = self._vectorize_sequences(x_train_raw, self.num_words)
        self.y_train = np.asarray(y_train_raw).astype('int32')  # Use int32 instead of float32

        self.x_test = self._vectorize_sequences(x_test_raw, self.num_words)
        self.y_test = np.asarray(y_test_raw).astype('int32')  # Use int32 instead of float32
        print("[INFO] Data prepared.")

    def build_model(self):
        print("[INFO] Building model...")
        self.model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(self.num_words,)),  # Define input shape
            layers.Dense(64, activation="relu"),
            layers.Dense(46, activation="softmax")
        ])
        
        # model compilation
        self.model.compile(optimizer="rmsprop",
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        print("[INFO] Model built.")

    def train(self, epochs=20, plot=False):
        if self.model is None:
            print('[INFO] Model is not built.')
            return
        
        print(f"[INFO] Training model for {epochs} epochs...")
        history = self.model.fit(self.x_train, self.y_train, 
                                 epochs=epochs, validation_split=0.2,
                                 batch_size=BATCH_SIZE, verbose=1)  # Added verbose=1 to show progress
        
        print("[INFO] Training complete.")
        
        if plot:
            self.plot_loss(history)
            self.plot_accuracy(history)
            
        return history

    def plot_loss(self, history):
        # Plotting the training and validation loss
        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.figure(1)
        plt.plot(epochs, loss_values, "r", label="Training loss")
        plt.plot(epochs, val_loss_values, "r--", label="Validation loss")
        plt.title("Reuters Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self, history):
        history_dict = history.history
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        epochs = range(1, len(acc) + 1)
        plt.figure(2)
        plt.plot(epochs, acc, "b", label="Training acc")
        plt.plot(epochs, val_acc, "b--", label="Validation acc")
        plt.title("Reuters Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def evaluate(self):
        if self.model is None:
            print('[INFO] Model is not built.')
            return
        
        print("[INFO] Evaluating model...")
        score = self.model.evaluate(self.x_test, self.y_test)
        print(f'[INFO] Test loss: {score[0]}')
        print(f'[INFO] Test accuracy: {score[1]}')

    def _vectorize_sequences(self, sequences, dimension):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            for j in sequence:
                results[i, j] = 1.
        return results

# Main execution
if __name__ == "__main__":
    print("[INFO] Starting the execution...")
    reuters_model = Reuters()
    reuters_model.prepare_data()
    reuters_model.build_model()
    history = reuters_model.train(epochs=5, plot=True)  # Reduced epochs to 5 for quicker testing
    reuters_model.evaluate()

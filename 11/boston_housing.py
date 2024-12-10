# Suppress TensorFlow informational logs and optional oneDNN optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress logs below warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations if needed

# Import required packages
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Define the Boston_Housing class
class Boston_Housing:
    def __init__(self, num_epochs=20, batch_size=16):
        self.model = None
        self.all_mae_histories = []
        self.epochs = num_epochs
        self.batch_size = batch_size
        self._load_and_normalize()

    def _load_and_normalize(self):
        # Downloading the dataset
        (self.x_train_raw, self.y_train_raw), (self.x_test_raw, self.y_test_raw) = boston_housing.load_data()

        # Data Normalization
        mean = self.x_train_raw.mean(axis=0)
        std = self.x_train_raw.std(axis=0)
        self.x_train_raw = (self.x_train_raw - mean) / std
        self.x_test_raw = (self.x_test_raw - mean) / std

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(self.x_train_raw.shape[1],)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        return model

    def k_fold_validations(self, k=4):
        num_val_samples = len(self.x_train_raw) // k
        for i in range(k):
            print(f"Processing fold #{i}")
            val_data = self.x_train_raw[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self.y_train_raw[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                [self.x_train_raw[:i * num_val_samples], self.x_train_raw[(i + 1) * num_val_samples:]],
                axis=0)
            partial_train_targets = np.concatenate(
                [self.y_train_raw[:i * num_val_samples], self.y_train_raw[(i + 1) * num_val_samples:]],
                axis=0)

            self.model = self._build_model()  # Reinitialize the model for each fold
            history = self.model.fit(
                partial_train_data, partial_train_targets,
                epochs=self.epochs,
                validation_data=(val_data, val_targets),
                batch_size=self.batch_size,
                verbose=0
            )
            mae_history = history.history.get("val_mae", history.history.get("val_mean_absolute_error"))
            self.all_mae_histories.append(mae_history)

    def plot_validation_mae(self):
        average_mae_history = [
            np.mean([x[i] for x in self.all_mae_histories]) for i in range(self.epochs)
        ]
        plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
        plt.title("Boston Housing Validation MAE")
        plt.xlabel("Epochs")
        plt.ylabel("Validation MAE")
        plt.show()

    def train(self, num_epochs):
        if self.model is None:
            self.model = self._build_model()
        history = self.model.fit(
            self.x_train_raw, self.y_train_raw,
            epochs=num_epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return history

    def evaluate(self):
        if self.model is None:
            print('[INFO] Model is not trained yet.')
            return
        score = self.model.evaluate(self.x_test_raw, self.y_test_raw, verbose=0)
        print(f'[INFO] Test loss: {score[0]}')
        print(f'[INFO] Test MAE: {score[1]}')

# Main block
if __name__ == "__main__":
    # Initialize the class
    boston_housing = Boston_Housing(num_epochs=20, batch_size=16)
    
    # Perform k-fold validation
    boston_housing.k_fold_validations(k=4)
    
    # Plot validation MAE
    boston_housing.plot_validation_mae()
    
    # Train the model with full training data
    print("Training the model on the entire dataset...")
    boston_housing.train(num_epochs=20)
    
    # Evaluate the model
    print("Evaluating the model on the test dataset...")
    boston_housing.evaluate()

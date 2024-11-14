import numpy as np
import pickle
from mnist import Mnist
from two_layer_net import TwoLayerNetWithBackProp

# Load the MNIST dataset
try:
    mnist = Mnist()
    (x_train, y_train), (x_test, y_test) = mnist.load()
except Exception as e:
    print(f"Error loading MNIST data: {e}")
    exit(1)

# Initialize the neural network
network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)

# Hyperparameters
epochs = 20            # Number of epochs to train
batch_size = 16
learning_rate = 0.01
train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)

train_losses = []
train_accs = []
test_accs = []

# Training loop
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(iter_per_epoch):
        # Create a batch of training data
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        # Calculate gradients using backpropagation
        grads = network.gradient(x_batch, y_batch)

        # Update each parameter with gradient descent
        for key in ('w1', 'b1', 'w2', 'b2'):
            network.params[key] -= learning_rate * grads[key]

        # Record the loss for each batch
        loss = network.loss(x_batch, y_batch)
        train_losses.append(loss)
        epoch_loss += loss

    # Calculate and store accuracy after each epoch
    train_acc = network.accuracy(x_train, y_train)
    test_acc = network.accuracy(x_test, y_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    avg_epoch_loss = epoch_loss / iter_per_epoch

    # Print epoch summary
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# Save the entire trained model to a pickle file
model_filename = "RishiD_weights.pkl"
network.update_layers()  # Finalize the model if needed
try:
    with open(model_filename, 'wb') as f:
        pickle.dump(network, f)
    print(f"Entire model saved as {model_filename}")
except Exception as e:
    print(f"Error saving the model: {e}")

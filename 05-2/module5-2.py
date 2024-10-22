import argparse
import matplotlib.pyplot as plt
import numpy as np
from mnist_data import MnistData

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Display an image from the MNIST dataset")
    parser.add_argument('dataset_type', choices=['train', 'test'], help="Specify 'train' or 'test' dataset.")
    parser.add_argument('index', type=int, help="Index of the image to display.")

    # Parse the arguments
    args = parser.parse_args()  # Argparser used for passing input arguments
    dataset_type = args.dataset_type
    index = args.index

    # Load the MNIST dataset
    mnist = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist.load()

    # Select the appropriate dataset based on user input
    images = train_images if dataset_type == 'train' else test_images
    labels = train_labels if dataset_type == 'train' else test_labels

    # Check if the index is valid
    if index < 0 or index >= len(images):
        print(f"Error: Index {index} is out of range for {dataset_type} dataset.")
        return

    # Reshape the image for display
    image = images[index].reshape(28, 28)
    label = np.argmax(labels[index])  # Assuming labels are one-hot encoded

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

    # Print the label of the displayed image
    print(f'The label for the {dataset_type} image at index {index} is: {label}')

# Entry point of the script
if __name__ == '__main__':
    main()

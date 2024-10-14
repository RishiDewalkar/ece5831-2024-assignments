import cv2
import numpy as np
import tensorflow as tf

# Function to load class names from labels.txt
def load_labels(label_file):
    """
    Load class names from the labels.txt file.

    This function reads a text file containing the class names (rock, paper, scissors),
    one per line, and returns them as a list.

    Args:
    - label_file (str): Path to the labels.txt file.

    Returns:
    - class_names (list of str): A list of class names (rock, paper, scissors).
    """
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Open Teachable Machine and load the trained model.
model = tf.keras.models.load_model('model/keras_model.h5')

# Open the labels.txt file and load the class names.
class_names = load_labels('model/labels.txt')

# Set up the webcam.
cap = cv2.VideoCapture(0)

while True:
    # Main loop for capturing video frames and making predictions in real-time.
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize and normalise the image in preparation for model prediction.
    img = cv2.resize(frame, (224, 224))  # Resize to model's input size
    img = np.array(img, dtype=np.float32) / 255.0  
    # Add an extra dimension to the array to represent the batch size (required by the model)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    #  Predict the class of the image using the model
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    prediction_label = class_names[class_idx]
    confidence = predictions[0][class_idx] * 100  # Convert to percentage

    #  Display the resulting frame with the predicted class label
    text = f"{prediction_label} {confidence:.2f}%"  # Format the text
    cv2.putText(
        frame,  # The original frame (NumPy array)
        text,  # The text to display
        (50, 50),  # Position on the frame (x, y)
        cv2.FONT_HERSHEY_COMPLEX,  # Font type
        1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Rock Paper Scissors', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows when the loop is finished
cap.release()
cv2.destroyAllWindows()
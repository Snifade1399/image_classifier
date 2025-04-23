import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Dataset directory
dataset_dir = '/home/snifade/Project/image_classification_project/friends_animals_datasets'

# Ensure the class names are correctly ordered
class_names = sorted(os.listdir(dataset_dir))  
print("Class names:", class_names)  # Print class names for verification

# Load the trained model
model = tf.keras.models.load_model('friends_animals_face_model.keras')

# Load an image to test
img_path = '/home/snifade/Downloads/naveen_test.jpg'  # Replace with the actual path to your test image
img = image.load_img(img_path, target_size=(224, 224))  # Resize to the input size of the model
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add an extra batch dimension

# Normalize the image (same as during training)
img_array /= 255.0

# Make prediction
predictions = model.predict(img_array)

# Print raw predictions to check what the model outputs
print("Raw predictions:", predictions)

# Get the predicted class label (max probability)
predicted_class = np.argmax(predictions, axis=1)
print("Predicted class index:", predicted_class[0])  # Print the predicted class index

# Check if the predicted index is within the valid range
if predicted_class[0] < len(class_names):
    print(f"Predicted class: {class_names[predicted_class[0]]}")
else:
    print("Error: Predicted class index is out of range.")
print("Raw predictions:", predictions)


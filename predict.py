import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Dataset directory
dataset_dir = '/home/snifade/Project/image_classification_project/friends_animals_datasets'

# Get class names by checking only valid directories (filtering out files like .DS_Store)
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
print("Class names:", class_names)

# Load the trained model
model = tf.keras.models.load_model('friends_animals_face_model.keras')

# Load and preprocess image
img_path = '/home/snifade/Downloads/test_image.jpg'  # Replace with your test image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict
predictions = model.predict(img_array)
print("Raw predictions:", predictions)

predicted_class = np.argmax(predictions, axis=1)[0]
print("Predicted class index:", predicted_class)

# Safety check
if predicted_class >= len(class_names):
    print("⚠️ Error: Predicted class index is out of range.")
else:
    print(f"✅ Predicted class: {class_names[predicted_class]}")


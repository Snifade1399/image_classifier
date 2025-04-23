from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained MobileNet model
model = tf.keras.applications.MobileNet(weights='imagenet')

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload and classification
@app.route('/', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file part'
    
    file = request.files['image']
    
    if file.filename == '':
        return 'No selected file'
    
    # Save the file to the static folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Read the image
    img = Image.open(file_path)
    
    # Convert image to RGB (if not already in RGB format)
    img = img.convert('RGB')
    
    # Resize to 224x224 for MobileNet
    img = img.resize((224, 224))
    
    # Convert image to array
    img_array = np.array(img)
    
    # Add batch dimension and ensure it has 3 color channels
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNet

    
    # Convert image to array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNet
    
    # Get predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=10)[0]
    
    # Prepare predictions for rendering
    result = []
    for (imagenet_id, label, score) in decoded_predictions:
        result.append({
            'label': label,
            'probability': score * 100
        })
    
    return render_template('index.html', predictions=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
CLIENT_SECRET_FILE = 'credentials.json'  # Replace this with your actual path to credentials.json


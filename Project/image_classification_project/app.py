from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = MobileNetV2(weights='imagenet')

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    label = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename = img_file.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(img_path)

            # Predict
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array)
            decoded_preds = decode_predictions(preds, top=1)[0]
            label = decoded_preds[0][1].replace('_', ' ').title()
            confidence = round(decoded_preds[0][2] * 100, 2)
            image_path = filename

    return render_template('index.html', label=label, confidence=confidence, image_path=image_path)


# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)


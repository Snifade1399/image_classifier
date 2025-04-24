from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import pickle
from deepface import DeepFace
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

# === Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load MobileNet
mobilenet_model = tf.keras.applications.MobileNet(weights='imagenet')

# Load stored face embeddings
with open('friends_embeddings.pkl', 'rb') as f:
    known_embeddings = pickle.load(f)  # Dictionary: { "name": embedding_vector }

def find_closest_face(embedding, threshold=0.6):
    """Return the closest matching friend name or None if no match found."""
    min_dist = float('inf')
    match_name = None
    for name, known_emb in known_embeddings.items():
        dist = np.linalg.norm(embedding - known_emb)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            match_name = name
    return match_name

@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = None
    recognized_person = None
    file_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            print("[INFO] Saved file at:", file_path)
            print("[INFO] File size:", os.path.getsize(file_path), "bytes")

            # Try face recognition
            try:
                print("[INFO] Running DeepFace face recognition...")
                analysis = DeepFace.represent(img_path=file_path, model_name="ArcFace", enforce_detection=False)[0]
                embedding = np.array(analysis["embedding"])
                recognized_person = find_closest_face(embedding)
                print("[INFO] Recognized person:", recognized_person)
            except Exception as e:
                print("[ERROR] Face recognition failed:", str(e))
                recognized_person = None

            # Fallback to standard MobileNet classification
            if recognized_person is None:
                try:
                    print("[INFO] Running MobileNet classification...")
                    img = Image.open(file_path).convert('RGB').resize((224, 224))
                    img_array = np.expand_dims(np.array(img), axis=0)
                    img_array = preprocess_input(img_array)
                    preds = mobilenet_model.predict(img_array)
                    predictions = decode_predictions(preds, top=5)[0]
                    print("[INFO] MobileNet predictions:", predictions)
                except Exception as e:
                    print("[ERROR] MobileNet classification failed:", str(e))
                    predictions = None

    return render_template("index.html", 
                           predictions=predictions,
                           recognized_person=recognized_person,
                           image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)


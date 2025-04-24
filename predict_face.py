from deepface import DeepFace
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle  # Added for loading pickle files

def extract_embedding(image_path):
    # Extract embedding from an image using ArcFace (better than FaceNet for your case)
    embeddings = DeepFace.represent(image_path, model_name="ArcFace", enforce_detection=False)
    return embeddings[0]["embedding"]

def load_embeddings(file_name="friends_embeddings.pkl"):  # Change to .pkl file
    # Load saved embeddings from pickle file
    with open(file_name, 'rb') as f:
        return pickle.load(f)  # Load using pickle

def predict_person(image_path, embeddings):
    # Extract and normalize embedding of the input image
    input_embedding = extract_embedding(image_path)
    input_embedding = normalize([input_embedding])[0]

    best_match = None
    highest_score = -1

    for name, stored_embedding in embeddings.items():
        stored_embedding = normalize([stored_embedding])[0]
        score = cosine_similarity([input_embedding], [stored_embedding])[0][0]

        if score > highest_score:
            best_match = name
            highest_score = score

    # Threshold for unknown face
    if highest_score < 0.75:
        return "Unknown", highest_score

    return best_match, highest_score

if __name__ == "__main__":
    embeddings = load_embeddings()  # Load the embeddings saved as .pkl
    test_image_path = '/home/snifade/Downloads/test_image.jpg'  # 🔁 Replace with your test image
    predicted_person, score = predict_person(test_image_path, embeddings)
    print(f"Predicted: {predicted_person} | Similarity score: {score}")


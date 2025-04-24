import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
import os
import numpy as np
import pickle
from deepface import DeepFace

DATASET_DIR = "friends_animals_datasets"
OUTPUT_FILE = "friends_embeddings.pkl"
MODEL_NAME = "ArcFace"

embeddings = {}

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    person_embeddings = []
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        try:
            embedding_obj = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False)[0]
            person_embeddings.append(embedding_obj["embedding"])
        except Exception as e:
            print(f"[!] Failed to process {img_path}: {e}")

    if person_embeddings:
        avg_embedding = np.mean(person_embeddings, axis=0)
        embeddings[person_name] = avg_embedding
        print(f"[+] Added {person_name} with {len(person_embeddings)} embeddings.")

# Save to pickle
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"\n✅ Done. Saved to {OUTPUT_FILE}")


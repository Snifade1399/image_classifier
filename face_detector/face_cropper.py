import cv2
import os
from mtcnn import MTCNN

# Paths
input_dir = "/home/snifade/Downloads/ankit_dataset"  # folder with group photos
output_dir = os.path.expanduser("~/Project/image_classification_project/ankit_dataset")
os.makedirs(output_dir, exist_ok=True)

# Initialize MTCNN face detector
detector = MTCNN()

for filename in os.listdir(input_dir):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)

    # Resize the image to reduce memory usage
    img = cv2.resize(img, (640, 480))  # Resize to 640x480, you can adjust this if needed
    
    # Detect faces
    results = detector.detect_faces(img)
    
    if len(results) == 0:
        print(f"No faces found in {filename}. Skipping...")
        continue
    
    # Log detected faces and save crops
    for i, result in enumerate(results):
        x, y, w, h = result['box']
        print(f"Detected face {i}: Coordinates: (x={x}, y={y}, w={w}, h={h})")
        
        # Crop face from image
        face_crop = img[y:y+h, x:x+w]
        
        # Save cropped face
        out_path = os.path.join(output_dir, f"{filename}_face{i}.jpg")
        cv2.imwrite(out_path, face_crop)
        print(f"Saved cropped face: {out_path}")


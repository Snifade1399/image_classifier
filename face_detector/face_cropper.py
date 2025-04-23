import cv2
import os

# Paths
input_dir = "/home/snifade/Project/image_classification_project/ankit"  # folder with group photos
output_dir = os.path.expanduser("~/Project/image_classification_project/ankit_cropped")
os.makedirs(output_dir, exist_ok=True)

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for filename in os.listdir(input_dir):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"Detected {len(faces)} faces in {filename}")

    # Check if faces are detected
    if len(faces) == 0:
        print(f"No faces found in {filename}. Skipping...")
        continue

    # Logging each face's details instead of showing
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = img[y:y+h, x:x+w]
        print(f"Face {i}: Coordinates: (x={x}, y={y}, w={w}, h={h})")

        # Save each cropped face directly (without displaying)
        out_path = os.path.join(output_dir, f"{filename}_face{i}.jpg")
        cv2.imwrite(out_path, face_crop)
        print(f"Saved cropped face: {out_path}")

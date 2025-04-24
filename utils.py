from deepface import DeepFace

def is_human_face(img_path):
    try:
        objs = DeepFace.extract_faces(img_path, enforce_detection=False)
        return len(objs) > 0
    except:
        return False


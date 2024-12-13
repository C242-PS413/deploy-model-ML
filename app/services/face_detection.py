import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency with the first code
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Process the first detected face
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]  # Crop the face
    
    # Resize to (48, 48)
    face_resized = cv2.resize(face, (48, 48))
    
    # Normalize pixel values to range [0, 1]
    face_normalized = face_resized / 255.0

    return original, gray, face, face_resized  # Return original, grayscale, cropped face, and resized face

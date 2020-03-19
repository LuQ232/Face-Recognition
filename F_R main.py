import face_recognition
import os
import cv2
DATA_DIR = "data"
KNOWN_FACES_DIR="known_faces"
UNKNOWN_FACES_DIR="unknown_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{DATA_DIR}/{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{DATA_DIR}/{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)
        known_faces.append(encoding)
        known_names.append(name)

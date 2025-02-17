import os
import numpy as np
import pickle
from keras_facenet import FaceNet
from mtcnn import MTCNN
import cv2

# Load FaceNet model
embedder = FaceNet()
detector = MTCNN()

# Directory containing training images
dataset_path = "training_data"

face_embeddings = []
labels = []

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    
    if os.path.isdir(person_path):
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)

            # Convert to RGB
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face
            faces = detector.detect_faces(rgb_img)

            if faces:
                x, y, width, height = faces[0]['box']
                face = rgb_img[y:y+height, x:x+width]

                # Get embedding
                embedding = embedder.embeddings([face])[0]

                face_embeddings.append(embedding)
                labels.append(person_name)

# Save trained data
with open('saved_model/face_recognition.pkl', 'wb') as f:
    pickle.dump({'embeddings': face_embeddings, 'labels': labels}, f)

print("Model trained successfully!")

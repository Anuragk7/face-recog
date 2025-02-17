import cv2
import numpy as np
import pickle
import os
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# Load trained model
with open('saved_model/face_recognition.pkl', 'rb') as f:
    model_data = pickle.load(f)

face_embeddings = np.array(model_data['embeddings'])
labels = np.array(model_data['labels'])

# Load FaceNet and MTCNN
embedder = FaceNet()
detector = MTCNN()

# Initialize webcam
vid_cam = cv2.VideoCapture(0)

# Directory for storing temp frames
temp_dir = "temp_frames"
os.makedirs(temp_dir, exist_ok=True)

prev_frame = None
motion_detected = False
start_time = None

while True:
    ret, frame = vid_cam.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        prev_frame = gray_frame
        continue
    
    frame_diff = cv2.absdiff(prev_frame, gray_frame)
    prev_frame = gray_frame
    
    if np.mean(frame_diff) > 5:  # Motion threshold
        print('motion')
        start_time = time.time()
        motion_detected = True
        temp_path = os.path.join(temp_dir, f"frame_{int(time.time())}.jpg")
        cv2.imwrite(temp_path, frame)
    
        # Process stored frames
     # Delete temp file after processing
    else:
         for temp_file in os.listdir(temp_dir):
            temp_path = os.path.join(temp_dir, temp_file)
            stored_frame = cv2.imread(temp_path)
            rgb_frame = cv2.cvtColor(stored_frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)

            for face in faces:
                x, y, width, height = face['box']
                x, y = max(0, x), max(0, y)
                face_img = rgb_frame[y:y+height, x:x+width]
                embedding = embedder.embeddings([face_img])[0]
                
                min_distance = float("inf")
                identity = "Unknown"
                confidence = 0.0

                for stored_embedding, label in zip(face_embeddings, labels):
                    distance = cosine(stored_embedding, embedding)
                    conf = (1 - distance) * 100  # Confidence score in percentage

                    if distance < 0.5 and distance < min_distance:
                        min_distance = distance
                        identity = label
                        confidence = conf
                
                if confidence > 70:
                    with open("face_log.txt", "a") as log_file:
                        log_file.write(f"{identity}, {time.strftime('%Y-%m-%d %H:%M:%S')}")

            os.remove(temp_path) 


    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid_cam.release()
cv2.destroyAllWindows()

# Clean up any remaining temp files
for temp_file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, temp_file))
os.rmdir(temp_dir)
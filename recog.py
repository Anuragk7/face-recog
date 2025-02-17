import cv2
import numpy as np
import pickle
import datetime
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

def log_recognition(identity, confidence):
    """Logs recognized faces with more than 70% confidence."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}, ID: {identity}, Confidence: {confidence:.2f}%\n"
    with open("recognition_log.txt", "a") as log_file:
        log_file.write(log_entry)

while True:
    # Capture frame
    _, frame = vid_cam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)  # Ensure valid values

        face_img = rgb_frame[y:y+height, x:x+width]

        # Get embedding
        embedding = embedder.embeddings([face_img])[0]

        # Compare with stored embeddings
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

        # Draw box and label with confidence
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        display_text = f"{identity} ({confidence:.2f}%)"
        cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Log if confidence is more than 70%
        if confidence > 70:
            log_recognition(identity, confidence)

    # Show frame
    cv2.imshow('Face Recognition', frame)
    
    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid_cam.release()
cv2.destroyAllWindows()

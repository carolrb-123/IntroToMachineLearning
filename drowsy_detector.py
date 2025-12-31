import cv2
import joblib
import numpy as np
import time
import winsound
import os
from collections import deque
from skimage.feature import hog

# =====================
# CONFIGURATION
# =====================
MODEL_PATH = "drowsy_model.pkl"
LOG_PATH = "drowsy_log.txt"
IMG_SIZE = 100
WARNING_THRESHOLD = 2.0   # seconds
CRITICAL_THRESHOLD = 5.0  # seconds
BUFFER_SIZE = 10  # Reduced for faster response

# Load model
print(f"Loading model from {MODEL_PATH}...")
try:
    model_data = joblib.load(MODEL_PATH)
    svm = model_data["svm"]
    pca = model_data["pca"]
    hog_params = model_data["hog_params"]
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def log_event(event_type, details=""):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] {event_type}: {details}\n")

def extract_single_hog_features(img):
    img = cv2.equalizeHist(img)
    img = img / 255.0
    features = hog(
        img,
        orientations=hog_params["orientations"],
        pixels_per_cell=hog_params["pixels_per_cell"],
        cells_per_block=hog_params["cells_per_block"],
        block_norm="L2-Hys",
        transform_sqrt=True
    )
    return features.reshape(1, -1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Optimization: Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prediction_buffer = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)
    current_state = "Open"
    last_voted_value = 0
    confidence = 0.5 # Persistent confidence
    faces = [] 
    frame_count = 0

    closed_start_time = None
    last_alarm_time = 0

    # Initialize CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print("Starting monitoring... Press 'q' to quit.")
    log_event("SESSION_START", "Monitoring started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process detection logic every 2nd frame
        if frame_count % 2 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = [v * 2 for v in faces[0]]
                
                face_roi = gray[y:y+h, x:x+w]
                face_roi = clahe.apply(face_roi) 
                
                face_img_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                features = extract_single_hog_features(face_img_resized)
                features_pca = pca.transform(features)
                probs = svm.predict_proba(features_pca)[0]
                confidence = probs[1] # Update persistent confidence
                last_voted_value = 1 if confidence > 0.65 else 0
            else:
                last_voted_value = 0
                confidence = 0.0 # Clear confidence if face lost
        
        prediction_buffer.append(last_voted_value)
        closed_ratio = sum(prediction_buffer) / len(prediction_buffer)
        
        # Instant override for very high confidence to feel "Real Time"
        if last_voted_value == 1 and confidence > 0.85:
            current_state = "Closed"
        elif last_voted_value == 0 and confidence < 0.25:
            current_state = "Open"
        else:
            # Fallback to buffer for borderline cases
            if current_state == "Closed":
                if closed_ratio < 0.40: current_state = "Open"
            else:
                if closed_ratio > 0.60: current_state = "Closed"

        if len(faces) > 0:
            (x, y, w, h) = [v * 2 for v in faces[0]]
            color = (0, 0, 255) if current_state == "Closed" else (0, 255, 0)
            label = f"Eyes: {current_state} ({closed_ratio:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        now = time.time()
        detected_closed = (current_state == "Closed")
        
        if detected_closed:
            if closed_start_time is None: closed_start_time = now
            duration = now - closed_start_time
            if duration >= CRITICAL_THRESHOLD:
                cv2.putText(frame, "!!! CRITICAL: EYES CLOSED !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                if now - last_alarm_time > 0.6:
                    winsound.Beep(1200, 400)
                    last_alarm_time = now
                    log_event("CRITICAL_ALARM", "Eyes closed for 5+ seconds.")
            elif duration >= WARNING_THRESHOLD:
                cv2.putText(frame, "WARNING: Eyes Closed", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                if now - last_alarm_time > 1.2:
                    winsound.Beep(800, 200)
                    last_alarm_time = now
        else:
            closed_start_time = None

        cv2.imshow('Smart Driver Assistant', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    log_event("SESSION_END", "Monitoring stopped.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

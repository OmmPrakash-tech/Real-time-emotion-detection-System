import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model/Emotion_detection_model_new.h5"
CASCADE_PATH = "haarcascade/haarcascade_frontalface_default.xml"

model = load_model(MODEL_PATH)

emotion_labels = [
    "Angry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
]

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    result = "No Face Detected"

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)

        emotion = emotion_labels[np.argmax(prediction)]

        result = emotion

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    return frame, result
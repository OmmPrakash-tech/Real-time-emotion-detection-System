import streamlit as st
import cv2
from detect import predict_emotion

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Real-Time Emotion Detection")

st.write(
    "Click Start Detection and allow camera access."
)

run = st.button("Start Detection")

FRAME_WINDOW = st.image([])

if run:

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            st.error("Camera not detected")
            break

        frame, emotion = predict_emotion(frame)

        FRAME_WINDOW.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        st.write(f"### Current Emotion: {emotion}")
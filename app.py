import streamlit as st
import cv2
from deepface import DeepFace

# Function to extract frames from the video
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Function to detect faces in frames
def detect_faces(frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_frames.append(frame[y:y+h, x:x+w])
    return face_frames

# Function to analyze emotion in the face frames
def analyze_emotion(face_frames):
    emotions = []
    for face_frame in face_frames:
        try:
            result = DeepFace.analyze(face_frame, actions=['emotion'])
            if isinstance(result, list):
                result = result[0]
            emotions.append(result.get('dominant_emotion', 'unknown'))
        except Exception as e:
            st.error(f"Error analyzing face: {e}")
            emotions.append('unknown')
    return emotions

# Confidence level based on emotions
def determine_confidence_level(emotions):
    confidence_level = 0
    for emotion in emotions:
        if emotion == 'happy':
            confidence_level += 1
        elif emotion == 'neutral':
            confidence_level += 0.8
        elif emotion == 'surprise':
            confidence_level += 0.6
        elif emotion in ['sad', 'fear']:
            confidence_level += 0.4
        elif emotion in ['angry', 'disgust']:
            confidence_level += 0.1
    return confidence_level / len(emotions) if emotions else 0

# Streamlit interface
st.title("Video Emotion Analyzer")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save uploaded video file
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Analyzing video...")

    # Process the video
    frames = extract_frames(video_path)
    face_frames = detect_faces(frames)
    emotions = analyze_emotion(face_frames)
    confidence_level = determine_confidence_level(emotions)

    st.write(f"Detected emotions: {emotions}")
    st.write(f"Confidence Level: {round(confidence_level * 100, 2)}%")

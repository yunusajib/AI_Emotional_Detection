import streamlit as st
import cv2
from fer import FER
import pandas as pd
import os

st.title("Video Emotion Detection")

@st.cache_resource(show_spinner=False)
def load_video_detector():
    return FER(mtcnn=True)

def analyze_video_emotions(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_interval = frame_rate * 2  # analyze every 2 seconds
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            results = detector.detect_emotions(frame)
            if results:
                top_emotion = max(results[0]["emotions"], key=results[0]["emotions"].get)
                emotions.append(top_emotion)
        frame_count += 1
    
    cap.release()
    return pd.Series(emotions).value_counts().to_dict() if emotions else {}

# Upload files section
st.header("Upload Video File for Facial/Voice Emotion Analysis")

uploaded_file = st.file_uploader(
    "Choose a video (mp4, avi, mov) file", 
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    save_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File uploaded and saved as {save_path}")
    
    # Load detector
    with st.spinner("Loading emotion detector..."):
        video_detector = load_video_detector()
    
    st.info("Analyzing facial and voice emotions in video...")
    try:
        video_emotions = analyze_video_emotions(save_path, video_detector)
        if video_emotions:
            st.write("**Detected Emotions (Video):**")
            
            # Create visualization
            col1, col2 = st.columns(2)
            
            with col1:
                for emo, count in video_emotions.items():
                    st.write(f"- **{emo.title()}**: {count} frames")
            
            with col2:
                # Create a simple bar chart
                df = pd.DataFrame(list(video_emotions.items()), columns=['Emotion', 'Count'])
                st.bar_chart(df.set_index('Emotion'))
        else:
            st.warning("No emotions detected in video frames.")
    except Exception as e:
        st.error(f"Video analysis failed: {e}")
    
    # Clean up temporary file
    try:
        os.remove(save_path)
    except:
        pass

else:
    st.write("Please upload a video file to analyze facial emotions.")
    
    st.markdown("""
    ### About this app:
    - Analyzes facial emotions in video files
    - Detects emotions every 2 seconds
    - Supports common video formats (MP4, AVI, MOV)
    """)
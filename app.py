import streamlit as st
import cv2
import pandas as pd
import os
import tempfile
import traceback
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Video Emotion Detection",
    page_icon="😊",
    layout="wide"
)

st.title("Video Emotion Detection")

# Initialize session state
if 'detector_loaded' not in st.session_state:
    st.session_state.detector_loaded = False
    st.session_state.detector = None

@st.cache_resource(show_spinner=False)
def load_video_detector():
    """Load FER detector with error handling"""
    try:
        from fer import FER
        detector = FER(mtcnn=True)
        return detector
    except ImportError as e:
        st.error(f"Failed to import FER: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to initialize FER detector: {e}")
        return None

def analyze_video_emotions(video_path, detector):
    """Analyze emotions in video with robust error handling"""
    if detector is None:
        return {}
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        # Check if video opened successfully
        if not cap.isOpened():
            st.error("Could not open video file")
            return {}
        
        emotions = []
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Handle cases where frame rate detection fails
        if frame_rate <= 0:
            frame_rate = 30
            
        frame_interval = max(1, frame_rate * 2)  # analyze every 2 seconds
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                status_text.text(f"Analyzing frame {frame_count}/{total_frames}")
                
                try:
                    results = detector.detect_emotions(frame)
                    if results and len(results) > 0:
                        top_emotion = max(results[0]["emotions"], key=results[0]["emotions"].get)
                        emotions.append(top_emotion)
                except Exception as e:
                    st.warning(f"Error analyzing frame {frame_count}: {e}")
                    continue
            
            frame_count += 1
            
            # Update progress
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        return pd.Series(emotions).value_counts().to_dict() if emotions else {}
        
    except Exception as e:
        st.error(f"Error during video analysis: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return {}

def main():
    # Check if detector can be loaded
    if not st.session_state.detector_loaded:
        with st.spinner("Loading emotion detection model..."):
            st.session_state.detector = load_video_detector()
            st.session_state.detector_loaded = True
    
    if st.session_state.detector is None:
        st.error("Failed to load emotion detector. Please check the requirements and try again.")
        st.info("Common issues:")
        st.info("- Missing dependencies")
        st.info("- Incompatible package versions")
        st.info("- Memory limitations")
        return
    
    # Upload files section
    st.header("Upload Video File for Facial Emotion Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a video (mp4, avi, mov) file", 
        type=["mp4", "avi", "mov"],
        help="Upload a video file to analyze facial emotions"
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        # Display file info
        file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)  # MB
        st.info(f"File size: {file_size:.2f} MB")
        
        if st.button("Analyze Video"):
            st.info("Analyzing facial emotions in video... This may take a few minutes.")
            
            try:
                video_emotions = analyze_video_emotions(temp_path, st.session_state.detector)
                
                if video_emotions:
                    st.success("Analysis completed!")
                    st.write("**Detected Emotions:**")
                    
                    # Create visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Emotion Counts")
                        total_detections = sum(video_emotions.values())
                        for emo, count in video_emotions.items():
                            percentage = (count / total_detections) * 100
                            st.write(f"- **{emo.title()}**: {count} detections ({percentage:.1f}%)")
                    
                    with col2:
                        st.subheader("Emotion Distribution")
                        # Create a simple bar chart
                        df = pd.DataFrame(list(video_emotions.items()), columns=['Emotion', 'Count'])
                        st.bar_chart(df.set_index('Emotion'))
                        
                    # Dominant emotion
                    dominant_emotion = max(video_emotions, key=video_emotions.get)
                    st.success(f"**Dominant emotion detected**: {dominant_emotion.title()}")
                    
                else:
                    st.warning("No faces or emotions detected in the video. Try uploading a video with clear facial expressions.")
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.error("Please try with a different video file or check the file format.")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    else:
        st.write("Please upload a video file to analyze facial emotions.")
        
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            ### How it works:
            - **Facial Detection**: Uses MTCNN for face detection
            - **Emotion Recognition**: Analyzes facial expressions using FER (Facial Emotion Recognition)
            - **Sampling**: Analyzes frames every 2 seconds for efficiency
            - **Supported Formats**: MP4, AVI, MOV
            
            ### Tips for best results:
            - Use videos with clear, well-lit faces
            - Ensure faces are not too small in the frame
            - Videos with multiple people will analyze all detected faces
            - Shorter videos (< 5 minutes) process faster
            """)

if __name__ == "__main__":
    main()
from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import tempfile
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import numpy as np
from database import DetectionHistory, SessionLocal
import time
from collections import deque
import threading

model_yolo = None

# Global variables for detection tracking
current_detections = []
detection_history = deque(maxlen=50)
detection_lock = threading.Lock()

def load_model (model_path=settings.DETECTION_MODEL):
    global model_yolo
    if model_yolo is None:
        try:
            model_yolo = YOLO(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            model_yolo = None
    return model_yolo

def _display_detected_frames(conf, model, st_frame, image):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

class VideoProcessorBISINDO(VideoProcessorBase):
    def __init__(self, confidence, model):
        self.confidence = confidence
        self.model = model

    def recv(self, frame):
        global current_detections, detection_history, detection_lock
        
        image = frame.to_ndarray(format="bgr24")

        # Predict the objects in the image using the YOLOv8 model
        res = self.model.predict(image, conf=self.confidence)

        # Plot the detected objects on the video frame
        res_plotted = res[0].plot()
        
        # Extract detection information
        boxes = res[0].boxes
        current_frame_detections = []
        
        if boxes:
            for box in boxes:
                class_id = int(box.cls)
                class_name = self.model.names[class_id]
                conf_value = float(box.conf)
                current_frame_detections.append({
                    'name': class_name,
                    'confidence': conf_value,
                    'time': time.time()
                })
        
        # Thread-safe update of global detection variables
        with detection_lock:
            current_detections = current_frame_detections
            if current_frame_detections:
                detection_history.extend(current_frame_detections)
        
        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

class VideoProcessor(VideoProcessorBase):
    def __init__(self, confidence, model):
        self.confidence = confidence
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Predict the objects in the image using the YOLOv8 model
        res = self.model.predict(image, conf=self.confidence)

        # Plot the detected objects on the video frame
        res_plotted = res[0].plot()
        
        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

def display_detection_text():
    """Display current detections and history below webcam"""
    global current_detections, detection_history, detection_lock
    
    # Create containers for detection display
    detection_container = st.container()
    
    with detection_container:
        st.markdown("### 🤟 Real-time BISINDO Hand Sign Detection")
        
        # Display current detections
        current_col, history_col = st.columns([1, 1])
        
        with current_col:
            st.markdown("**Currently Detected:**")
            current_placeholder = st.empty()
            
        with history_col:
            st.markdown("**Recent History:**")
            history_placeholder = st.empty()
        
        # Control buttons
        control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
        
        with control_col1:
            if st.button("🔄 Refresh", key="refresh_detection"):
                st.rerun()
                
        with control_col2:
            if st.button("🗑️ Clear History", key="clear_history"):
                with detection_lock:
                    detection_history.clear()
                    current_detections.clear()
                st.success("History cleared!")
                
        with control_col3:
            show_confidence = st.checkbox("Show Confidence", value=True)
        
        # Display current detections
        with current_placeholder.container():
            with detection_lock:
                if current_detections:
                    for detection in current_detections:
                        confidence_color = get_confidence_color(detection['confidence'])
                        confidence_text = f" - {detection['confidence']:.2f}" if show_confidence else ""
                        st.markdown(f"{confidence_color} **{detection['name'].upper()}**{confidence_text}")
                else:
                    st.info("👋 Show a hand sign to detect...")
        
        # Display recent history
        with history_placeholder.container():
            with detection_lock:
                if detection_history:
                    # Get unique recent detections (last 10 seconds)
                    current_time = time.time()
                    recent_detections = [d for d in detection_history if current_time - d['time'] <= 10]
                    
                    # Group by sign name and show most recent with highest confidence
                    sign_groups = {}
                    for detection in recent_detections:
                        sign_name = detection['name']
                        if sign_name not in sign_groups or detection['confidence'] > sign_groups[sign_name]['confidence']:
                            sign_groups[sign_name] = detection
                    
                    if sign_groups:
                        for sign_name, detection in sign_groups.items():
                            time_ago = current_time - detection['time']
                            confidence_text = f" ({detection['confidence']:.2f})" if show_confidence else ""
                            st.write(f"• {sign_name.upper()}{confidence_text} - {time_ago:.1f}s ago")
                    else:
                        st.write("No recent history")
                else:
                    st.write("No history yet")

def get_confidence_color(confidence):
    """Return emoji color based on confidence level"""
    if confidence > 0.8:
        return "🟢"  # Green - Very confident
    elif confidence > 0.6:
        return "🟡"  # Yellow - Moderately confident
    elif confidence > 0.4:
        return "🟠"  # Orange - Low confidence
    else:
        return "🔴"  # Red - Very low confidence

def play_webcam_bisindo(conf, model):
    """Enhanced webcam function with BISINDO detection text display"""
    
    st.markdown("### 📹 BISINDO Hand Sign Detection")
    
    # WebRTC configuration
    webrtc_ctx = webrtc_streamer(
        key="bisindo_webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=lambda: VideoProcessorBISINDO(conf, model),
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence = conf
        webrtc_ctx.video_processor.model = model
    
    # Add some spacing
    st.markdown("---")
    
    # Display detection text below webcam
    display_detection_text()
    
    # Additional features
    with st.expander("📊 Detection Statistics"):
        with detection_lock:
            if detection_history:
                total_detections = len(detection_history)
                unique_signs = len(set(d['name'] for d in detection_history))
                avg_confidence = sum(d['confidence'] for d in detection_history) / len(detection_history)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Detections", total_detections)
                with col2:
                    st.metric("Unique Signs", unique_signs)
                with col3:
                    st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
            else:
                st.info("No detections yet. Start showing hand signs!")

def play_webcam(conf, model):
    """Original webcam function - now redirects to BISINDO version"""
    play_webcam_bisindo(conf, model)



def play_stored_video(conf, model):
    source_vid = st.sidebar.file_uploader("Choose a video...", type=("mp4", "avi", "mov", "mkv"))

    if source_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(source_vid.read())
        vid_cap = cv2.VideoCapture(tfile.name)

        st.video(tfile.name)

        if st.sidebar.button('Detect Objects'):
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break
    else:
        st.warning("Please upload a video file.")

def save_detection(source_type, source_path, detected_image):
    db = SessionLocal()
    new_record = DetectionHistory(
        source_type=source_type,
        source_path=source_path,
        detected_image=detected_image
    )
    db.add(new_record)
    db.commit()
    db.close()

def get_detection_history():
    db = SessionLocal()
    history = db.query(DetectionHistory).all()
    db.close()
    return history

def delete_detection_record(record_id):
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        record = session.query(DetectionHistory).get(record_id)
        if record:
            session.delete(record)
            session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
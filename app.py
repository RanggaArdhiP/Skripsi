# Import modules
from pathlib import Path
import PIL
import tempfile
import cv2
import base64
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Hand Sign Detection using YOLOv8",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customizing the sidebar with background color
st.markdown("""
    <style>
        .custom-header {
            background-color: #CB4A16;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .custom-header h2 {
            color: white;
        }
        .history-image {
            width: 80%;
            max-width: 300px;
            height: 50%;
        }
        .main-title {
            font-size: 40px;
            font-weight: 700;
            color: #4B39EF;
            margin-bottom: 15px;
        }

        .subtext {
            font-size: 18px;
            line-height: 1.6;
            color: #333;
            margin-bottom: 25px;
        }

        .info-card {
            background-color: #eef2ff;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
        }

        .info-card h3 {
            color: #4B39EF;
            font-size: 22px;
            margin-bottom: 10px;
        }

        .caption {
            font-size: 16px;
            color: #555;
            text-align: center;
        }

        .stApp {
            background-color: #f9f9fc;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with custom header and different background color for the main content
st.sidebar.markdown('<div class="custom-header"><h2>🤟 | HandSign</h2></div>', unsafe_allow_html=True)

page = st.sidebar.selectbox("Select Page", ["🏠 | Home", "🔎 | Detection", "⌛ | History"], index=0, key='page_selector')

# Home Page
if page == "🏠 | Home":
    st.markdown("<div class='main-title'>Welcome to BISINDO Hand Sign Detection 🤟</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class='subtext'>
            This web-based application uses <strong>YOLOv11</strong> to detect <strong>Bahasa Isyarat Indonesia (BISINDO)</strong> 
            hand gestures from images, video, or webcam in real-time. Designed to empower inclusive communication for the Deaf community 
            and support educational tools through advanced AI technology.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='info-card'>
            <h3>📖 What’s on this App?</h3>
            <ul>
                <li><strong>Home</strong>: Introduction and project background.</li>
                <li><strong>Detection</strong>: Upload or capture an image/video to detect BISINDO signs.</li>
                <li><strong>History</strong>: Review previous detection logs and analyzed results.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🖼️ Example of Hand Sign Detection")

    col1, col2 = st.columns(2)

    with col1:
        default_image_path = str(settings.DEFAULT_IMAGE)
        default_image = PIL.Image.open(default_image_path)
        st.image(default_image_path, caption="🤲 Original Hand Sign Image", use_column_width=True)
    
    with col2:
        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
        default_detected_image = PIL.Image.open(default_detected_image_path)
        st.image(default_detected_image_path, caption="🧠 YOLOv11 Detected BISINDO Sign", use_column_width=True)



# Detection Page
elif page == "🔎 | Detection":
    st.title("🤟 BISINDO Hand Sign Detection using YOLOv11")

    st.sidebar.header("ML Model Config")
    confidence = float(st.sidebar.slider("Select Model Confidence (%)", 25, 100, 40)) / 100

    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.subheader("Image/Video Config")
    source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

    source_img = None
    source_vid = None

    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image", use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image", use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    try:
                        res = model.predict(uploaded_image, conf=confidence)
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption='Detected Image', use_column_width=True)

                        # Display detected BISINDO signs prominently
                        st.markdown("---")
                        st.markdown("### 🤟 Detected BISINDO Hand Signs:")
                        
                        if boxes:
                            detected_signs = []
                            for box in boxes:
                                class_id = int(box.cls)
                                class_name = model.names[class_id]
                                conf_value = float(box.conf)
                                detected_signs.append({
                                    'name': class_name,
                                    'confidence': conf_value
                                })
                            
                            # Sort by confidence (highest first)
                            detected_signs.sort(key=lambda x: x['confidence'], reverse=True)
                            
                            # Display each detected sign with color coding
                            for sign in detected_signs:
                                if sign['confidence'] > 0.8:
                                    st.success(f"🟢 **{sign['name'].upper()}** - Confidence: {sign['confidence']:.2f}")
                                elif sign['confidence'] > 0.6:
                                    st.warning(f"🟡 **{sign['name'].upper()}** - Confidence: {sign['confidence']:.2f}")
                                else:
                                    st.info(f"🟠 **{sign['name'].upper()}** - Confidence: {sign['confidence']:.2f}")
                            
                            # Create a sentence from detected signs
                            if len(detected_signs) > 0:
                                sentence = " ".join([sign['name'].upper() for sign in detected_signs])
                                st.markdown(f"**Detected Sequence:** {sentence}")
                        else:
                            st.info("👋 No BISINDO hand signs detected in this image")

                        # Save detection result
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                            PIL.Image.fromarray(res_plotted).save(tmpfile.name)
                            with open(tmpfile.name, "rb") as file:
                                detected_image = file.read()
                                helper.save_detection("Image", source_img.name, detected_image)

                        try:
                            with st.expander("📊 Detailed Detection Results"):
                                if boxes:
                                    for i, box in enumerate(boxes):
                                        class_id = int(box.cls)
                                        class_name = model.names[class_id]
                                        conf_value = float(box.conf)
                                        st.write(f"Detection {i+1}: **{class_name}** - Confidence: {conf_value:.4f}")
                                else:
                                    st.write("No objects detected.")
                        except Exception as ex:
                            st.error("Error processing detection results.")
                            st.error(ex)
                    except Exception as ex:
                        st.error("Error running detection.")
                        st.error(ex)

    elif source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)

    elif source_radio == settings.WEBCAM:
        # Enhanced webcam with BISINDO detection text
        helper.play_webcam_bisindo(confidence, model)


    else:
        st.error("Please select a valid source type!")

# History Page
elif page == "⌛ | History":
    st.title("Detection History")
    history = helper.get_detection_history()

    if not history:
        st.warning("No Detection History")
    else:
        for record in history:
            st.write(f"Source Type: {record.source_type}")
            st.write(f"Source Path: {record.source_path}")
            image_data = base64.b64encode(record.detected_image).decode('utf-8')
            st.markdown(f'<img class="history-image" src="data:image/png;base64,{image_data}" alt="Detected Image">', unsafe_allow_html=True)
            if st.button('Delete', key=f'delete_{record.id}'):
                helper.delete_detection_record(record.id)
                st.rerun()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from rembg import remove
import os
import json
from utils import (
    load_places_model, predict_scene,
    load_emotion_model, detect_emotions,
    analyze_face_location, generate_report,
    load_face_detector, detect_faces
)

# Set page config
st.set_page_config(
    page_title="AnubhavAI - Advanced Image Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .report-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .face-box {
        border: 1px solid #e0e0e0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .emotion-bar {
        background-color: #e0e0e0;
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .emotion-fill {
        background-color: #4CAF50;
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease-in-out;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤖 AnubhavAI - Advanced Image Analysis")
st.markdown("""
This application provides comprehensive image analysis capabilities including:
- Face Detection & Emotion Analysis
- Background Segmentation
- Scene Classification
- Detailed Analysis Report
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI-powered application analyzes images to provide:
    - Face detection and emotion analysis
    - Background removal
    - Scene classification
    - Detailed analysis report
    """)
    
    st.header("How to Use")
    st.markdown("""
    1. Upload an image using the file uploader
    2. Wait for the analysis to complete
    3. View the results in the main panel
    4. Download the analysis report
    """)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = {}

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Get image information
        image_info = {
            "filename": uploaded_file.name,
            "size": image.size,
            "mode": image.mode,
            "format": image.format
        }
        
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create columns for different features
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Face Detection & Emotion Analysis")
            try:
                # Load face detection models
                face_cascade, profile_cascade = load_face_detector()
                
                # Detect faces using both frontal and profile cascades
                faces = detect_faces(image_cv, face_cascade, profile_cascade)
                
                # Draw rectangle around faces and analyze emotions
                face_img = image_cv.copy()
                emotion_model = load_emotion_model()
                
                face_analysis = []
                for i, (x, y, w, h) in enumerate(faces):
                    # Draw rectangle
                    cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(face_img, f"Face {i+1}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Extract face region
                    face_region = image_cv[y:y+h, x:x+w]
                    
                    # Analyze face location
                    face_location = analyze_face_location(face_region, image_cv.shape[:2])
                    
                    # Detect emotions
                    emotions = detect_emotions(face_region, emotion_model)
                    face_analysis.append({
                        "face_id": i + 1,
                        "position": (x, y, w, h),
                        "location": face_location,
                        "emotions": emotions
                    })
                
                st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption="Face Detection")
                st.markdown(f"""
                <div class="detection-box">
                    <h4>Face Detection Results</h4>
                    <p>Total faces detected: {len(faces)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display emotion analysis for each face
                for face in face_analysis:
                    with st.expander(f"Face {face['face_id']} Analysis"):
                        st.markdown(f"**Location:** {face['location']['description']}")
                        st.markdown(f"**Size:** {face['location']['size_ratio']:.1f}% of image")
                        
                        for emotion in face['emotions']:
                            st.markdown(f"**{emotion['emotion']}**")
                            st.markdown(f"*{emotion['description']}*")
                            st.markdown(f"""
                            <div class="emotion-bar">
                                <div class="emotion-fill" style="width: {emotion['probability']}%"></div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.write(f"Confidence: {emotion['probability']:.2f}%")
                
            except Exception as e:
                st.error(f"Face detection failed: {str(e)}")
        
        with col2:
            st.subheader("Background Segmentation")
            try:
                # Remove background
                output = remove(image)
                st.image(output, caption="Background Removed")
            except Exception as e:
                st.error(f"Background removal failed: {str(e)}")
        
        # Scene Classification
        st.subheader("Scene Classification")
        try:
            scene_model = load_places_model()
            scene_predictions = predict_scene(image, scene_model)
            
            # Display predictions with progress bars
            for pred in scene_predictions:
                st.markdown(f"**{pred['category']}**")
                st.markdown(f"*{pred['description']}*")
                st.progress(pred['probability'] / 100)
                st.write(f"Confidence: {pred['probability']:.2f}%")
                
        except Exception as e:
            st.error(f"Scene classification failed: {str(e)}")
        
        # Generate and display detailed report
        report = generate_report(face_analysis, scene_predictions, image_info)
        st.session_state.analysis_report = report
        
        # Detailed Analysis Report
        st.markdown("### 📊 Detailed Analysis Report")
        report_container = st.container()
        
        with report_container:
            st.markdown("""
            <div class="report-box">
                <h4>Image Analysis Summary</h4>
                <ul>
                    <li>Total faces detected: {}</li>
                    <li>Primary scene category: {}</li>
                    <li>Background successfully removed: {}</li>
                    <li>Analysis timestamp: {}</li>
                </ul>
            </div>
            """.format(
                len(faces),
                scene_predictions[0]['category'] if scene_predictions else "Unknown",
                "Yes" if 'output' in locals() else "No",
                report['timestamp']
            ), unsafe_allow_html=True)
            
            # Face Analysis Details
            st.markdown("#### Face Analysis Details")
            for face in face_analysis:
                with st.expander(f"Face {face['face_id']} Details"):
                    st.markdown(f"**Position:** x={face['position'][0]}, y={face['position'][1]}")
                    st.markdown(f"**Location:** {face['location']['description']}")
                    st.markdown(f"**Size:** {face['location']['size_ratio']:.1f}% of image")
                    st.markdown("**Emotions detected:**")
                    for emotion in face['emotions']:
                        st.markdown(f"- {emotion['emotion']}: {emotion['probability']:.2f}%")
                        st.markdown(f"  *{emotion['description']}*")
            
            # Scene Analysis Details
            st.markdown("#### Scene Analysis Details")
            for pred in scene_predictions:
                st.markdown(f"**{pred['category']}**")
                st.markdown(f"*{pred['description']}*")
                st.write(f"Confidence: {pred['probability']:.2f}%")
        
        # Download options
        st.markdown("### 📥 Download Analysis Report")
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name="image_analysis_report.json",
            mime="application/json"
        )
        
        # Generate text report
        text_report = f"""Image Analysis Report
Generated on: {report['timestamp']}

Image Information:
- Filename: {image_info['filename']}
- Size: {image_info['size']}
- Format: {image_info['format']}

Face Analysis:
Total faces detected: {len(face_analysis)}

"""
        for face in face_analysis:
            text_report += f"""
Face {face['face_id']}:
- Position: x={face['position'][0]}, y={face['position'][1]}
- Location: {face['location']['description']}
- Size: {face['location']['size_ratio']:.1f}% of image
- Emotions:
"""
            for emotion in face['emotions']:
                text_report += f"  * {emotion['emotion']}: {emotion['probability']:.2f}%\n"
                text_report += f"    {emotion['description']}\n"
        
        text_report += f"""
Scene Analysis:
Primary scene: {scene_predictions[0]['category']}
Description: {scene_predictions[0]['description']}
Confidence: {scene_predictions[0]['probability']:.2f}%

Additional scene predictions:
"""
        for pred in scene_predictions[1:]:
            text_report += f"- {pred['category']}: {pred['probability']:.2f}%\n"
            text_report += f"  {pred['description']}\n"
        
        st.download_button(
            label="Download Report (TXT)",
            data=text_report,
            file_name="image_analysis_report.txt",
            mime="text/plain"
        )
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload an image to begin analysis.")
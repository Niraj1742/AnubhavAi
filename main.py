"""
AnubhavAI - Advanced Image Analysis Platform
Main Application File

This file contains the main Streamlit application that provides a web interface
for advanced image analysis capabilities including face detection, emotion analysis,
background removal, and scene classification.
"""

# Standard library imports
import os
import json

# Third-party imports
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from rembg import remove

# Local imports
from utils import (
    load_places_model, predict_scene,
    load_emotion_model, detect_emotions,
    analyze_face_location, generate_report,
    load_face_detector, detect_faces
)

# Configure Streamlit page settings
st.set_page_config(
    page_title="AnubhavAI - Advanced Image Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the application
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

# Application title and description
st.title("🤖 AnubhavAI - Advanced Image Analysis")
st.markdown("""
This application provides comprehensive image analysis capabilities including:
- Face Detection & Emotion Analysis
- Background Segmentation
- Scene Classification
- Detailed Analysis Report
""")

# Sidebar content
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

# Initialize session state variables
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = {}

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Extract image information
        image_info = {
            "filename": uploaded_file.name,
            "size": image.size,
            "mode": image.mode,
            "format": image.format
        }
        
        # Convert PIL Image to OpenCV format for processing
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create two columns for feature display
        col1, col2 = st.columns(2)
        
        # Face Detection and Emotion Analysis Column
        with col1:
            st.subheader("Face Detection & Emotion Analysis")
            try:
                # Load face detection models
                face_cascade, profile_cascade = load_face_detector()
                
                # Detect faces using both frontal and profile cascades
                faces = detect_faces(image_cv, face_cascade, profile_cascade)
                
                # Process each detected face
                face_img = image_cv.copy()
                emotion_model = load_emotion_model()
                
                face_analysis = []
                for i, (x, y, w, h) in enumerate(faces):
                    # Draw rectangle around face
                    cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(face_img, f"Face {i+1}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Extract face region for analysis
                    face_region = image_cv[y:y+h, x:x+w]
                    
                    # Analyze face location in image
                    face_location = analyze_face_location(face_region, image_cv.shape[:2])
                    
                    # Detect emotions in face
                    emotions = detect_emotions(face_region, emotion_model)
                    face_analysis.append({
                        "face_id": i + 1,
                        "position": (x, y, w, h),
                        "location": face_location,
                        "emotions": emotions
                    })
                
                # Display face detection results
                st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption="Face Detection")
                st.markdown(f"""
                <div class="detection-box">
                    <h4>Face Detection Results</h4>
                    <p>Total faces detected: {len(faces)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed emotion analysis for each face
                for face in face_analysis:
                    with st.expander(f"Face {face['face_id']} Analysis"):
                        st.markdown(f"**Location:** {face['location']['description']}")
                        st.markdown(f"**Size:** {face['location']['size_ratio']:.1f}% of image")
                        
                        # Display emotion probabilities
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
        
        # Background Removal Column
        with col2:
            st.subheader("Background Segmentation")
            try:
                # Remove background from image
                output = remove(image)
                st.image(output, caption="Background Removed")
            except Exception as e:
                st.error(f"Background removal failed: {str(e)}")
        
        # Scene Classification Section
        st.subheader("Scene Classification")
        try:
            # Load and run scene classification model
            scene_model = load_places_model()
            scene_predictions = predict_scene(image, scene_model)
            
            # Display scene classification results
            st.markdown("### Scene Analysis Results")
            for scene in scene_predictions:
                st.markdown(f"**{scene['scene']}**")
                st.markdown(f"*{scene['description']}*")
                st.markdown(f"""
                <div class="emotion-bar">
                    <div class="emotion-fill" style="width: {scene['probability']}%"></div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"Confidence: {scene['probability']:.2f}%")
        
        except Exception as e:
            st.error(f"Scene classification failed: {str(e)}")
        
        # Generate and display analysis report
        analysis_report = generate_report(face_analysis, scene_predictions, image_info)
        st.session_state.analysis_report = analysis_report
        
        # Download report button
        st.download_button(
            label="Download Analysis Report",
            data=json.dumps(analysis_report, indent=2),
            file_name="analysis_report.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload an image to begin analysis.")
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
import sys
import time
import logging
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

# Third-party imports
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page settings - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AnubhavAI - Advanced Image Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODELS_DIR = "models"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Custom CSS for styling the application
def load_css():
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-weight: bold;
    }
    .report-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .face-box {
        border: 1px solid #e0e0e0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        background-color: #ffffff;
    }
    .emotion-bar {
        background-color: #e9ecef;
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .emotion-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease-in-out;
    }
    .detection-box {
        border: 2px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        background-color: #ffffff;
    }
    .error-box {
        border: 2px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        background-color: #fff5f5;
    }
    .success-box {
        border: 2px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        background-color: #f8fff9;
    }
    .feature-card {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .progress-container {
        position: relative;
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .progress-bar {
        position: absolute;
        height: 100%;
        background-color: #007bff;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .progress-text {
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS
load_css()

class ImageAnalysisApp:
    """Main application class for AnubhavAI Image Analysis Platform."""

    def __init__(self):
        """Initialize the application."""
        self.face_detector = None
        self.emotion_model = None
        self.scene_model = None
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize all required models."""
        try:
            # Initialize face detector
            from utils import load_face_detector
            self.face_detector = load_face_detector()

            # Initialize emotion model
            from utils import load_emotion_model
            self.emotion_model = load_emotion_model()

            # Initialize scene model
            from utils import load_places_model
            self.scene_model = load_places_model()

            st.success("All models loaded successfully!")
        except Exception as e:
            st.error(f"Failed to initialize models: {str(e)}")
            logger.error(f"Model initialization error: {str(e)}")

    def run(self) -> None:
        """Run the Streamlit application."""
        self._display_header()
        self._display_sidebar()
        self._process_uploaded_image()

    def _display_header(self) -> None:
        """Display the application header."""
        st.title("🤖 AnubhavAI - Advanced Image Analysis")
        st.markdown("""
        <div class="feature-card">
            <h4>Welcome to AnubhavAI Image Analysis Platform</h4>
            <p>This application provides comprehensive image analysis capabilities including:</p>
            <ul>
                <li>👤 Face Detection & Emotion Analysis</li>
                <li>🎭 Background Segmentation</li>
                <li>🌄 Scene Classification</li>
                <li>📊 Detailed Analysis Report</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    def _display_sidebar(self) -> None:
        """Display the sidebar content."""
        with st.sidebar:
            st.header("About AnubhavAI")
            st.markdown("""
            <div class="feature-card">
                <p>AnubhavAI is an advanced image analysis platform that uses
                state-of-the-art computer vision techniques to provide detailed insights
                about your images.</p>
                <p>Key features include:</p>
                <ul>
                    <li>Multi-face detection with emotion analysis</li>
                    <li>Background removal with transparency support</li>
                    <li>Scene classification with confidence scores</li>
                    <li>Comprehensive analysis reports</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.header("How to Use")
            st.markdown("""
            <div class="feature-card">
                <ol>
                    <li>Upload an image using the file uploader</li>
                    <li>Wait for the analysis to complete</li>
                    <li>View the results in the main panel</li>
                    <li>Download the analysis report</li>
                </ol>
                <p>For best results, use clear images with visible faces.</p>
            </div>
            """, unsafe_allow_html=True)

            st.header("System Information")
            st.markdown(f"""
            <div class="feature-card">
                <p><strong>Python Version:</strong> {sys.version}</p>
                <p><strong>Python Executable:</strong> {sys.executable}</p>
            </div>
            """, unsafe_allow_html=True)

    def _process_uploaded_image(self) -> None:
        """Process the uploaded image."""
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image for analysis (JPG, JPEG, or PNG format)"
        )

        if uploaded_file is not None:
            try:
                # Show processing status
                with st.spinner("Processing image..."):
                    start_time = time.time()

                    # Read and display the uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                    # Convert PIL Image to OpenCV format for processing
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # Create tabs for different analysis sections
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Face Analysis",
                        "Background Removal",
                        "Scene Classification",
                        "Analysis Report"
                    ])

                    # Initialize session state for analysis results
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = {}

                    with tab1:
                        self._process_face_analysis(image, image_cv)

                    with tab2:
                        self._process_background_removal(image)

                    with tab3:
                        self._process_scene_classification(image)

                    with tab4:
                        self._display_analysis_report()

                    # Show processing time
                    processing_time = time.time() - start_time
                    st.success(f"Analysis completed in {processing_time:.2f} seconds")

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.error(f"Image processing error: {str(e)}", exc_info=True)

    def _process_face_analysis(self, image: Image.Image, image_cv: np.ndarray) -> None:
        """Process face detection and emotion analysis."""
        try:
            st.subheader("Face Detection & Emotion Analysis")

            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Face Detection
            status_text.text("Detecting faces...")
            progress_bar.progress(20)

            # Detect faces
            faces = self._detect_faces(image_cv)

            # Step 2: Emotion Analysis
            status_text.text("Analyzing emotions...")
            progress_bar.progress(50)

            # Process each detected face
            face_img = image_cv.copy()
            face_analysis = []

            for i, (x, y, w, h) in enumerate(faces):
                # Draw rectangle around face
                cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(face_img, f"Face {i+1}", (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Extract face region for analysis
                face_region = image_cv[y:y+h, x:x+w]

                # Analyze face location in image
                face_location = self._analyze_face_location(face_region, image_cv.shape[:2])

                # Detect emotions in face
                emotions = self._detect_emotions(face_region)
                face_analysis.append({
                    "face_id": i + 1,
                    "position": (x, y, w, h),
                    "location": face_location,
                    "emotions": emotions
                })

            # Step 3: Finalizing Results
            status_text.text("Finalizing results...")
            progress_bar.progress(100)

            # Display face detection results
            st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption="Face Detection Results")

            # Store results in session state
            st.session_state.analysis_results['face_analysis'] = face_analysis

            # Display summary
            st.markdown(f"""
            <div class="detection-box">
                <h4>Face Detection Results</h4>
                <p>Total faces detected: {len(faces)}</p>
            </div>
            """, unsafe_allow_html=True)

            # Display detailed emotion analysis for each face
            for face in face_analysis:
                with st.expander(f"Face {face['face_id']} Analysis", expanded=False):
                    st.markdown(f"**Location:** {face['location']['description']}")
                    st.markdown(f"**Size:** {face['location']['size_ratio']:.1f}% of image")

                    # Display emotion probabilities
                    for emotion in face['emotions']:
                        st.markdown(f"**{emotion['emotion']}**")
                        st.markdown(f"*{emotion['description']}*")
                        st.markdown(f"""
                        <div class="emotion-bar">
                            <div class="emotion-fill" style="width: {emotion['probability']}%; background-color: {'#28a745' if emotion['probability'] > 50 else '#ffc107'}"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(f"Confidence: {emotion['probability']:.2f}%")

            status_text.text("Face analysis completed!")

        except Exception as e:
            st.error(f"Face detection failed: {str(e)}")
            logger.error(f"Face analysis error: {str(e)}", exc_info=True)

    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image."""
        try:
            if self.face_detector is None:
                st.error("Face detector not initialized. Please check model loading.")
                return []
                
            from utils import detect_faces
            face_cascade, profile_cascade = self.face_detector
            return detect_faces(image, face_cascade, profile_cascade)
        except Exception as e:
            st.error(f"Error in face detection: {str(e)}")
            return []

    def _analyze_face_location(
        self,
        face_region: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Analyze the location of a face within the image."""
        try:
            from utils import analyze_face_location
            return analyze_face_location(face_region, image_size)
        except Exception as e:
            st.error(f"Error analyzing face location: {str(e)}")
            return {"description": "Unknown", "size_ratio": 0}

    def _detect_emotions(self, face_region: np.ndarray) -> List[Dict[str, Any]]:
        """Detect emotions in a face region."""
        try:
            from utils import detect_emotions
            return detect_emotions(face_region, self.emotion_model)
        except Exception as e:
            st.error(f"Error detecting emotions: {str(e)}")
            return []

    def _process_background_removal(self, image: Image.Image) -> None:
        """Process background removal."""
        try:
            st.subheader("Background Segmentation")

            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Removing background...")
            progress_bar.progress(50)

            # Convert PIL Image to numpy array
            input_array = np.array(image)
            
            # Create a mask for GrabCut
            mask = np.zeros(input_array.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define rectangle for foreground (adjust these values based on your image)
            rect = (10, 10, input_array.shape[1]-10, input_array.shape[0]-10)
            
            # Apply GrabCut
            cv2.grabCut(input_array, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create mask where sure and likely fg
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            
            # Apply mask to image
            output_array = input_array * mask2[:,:,np.newaxis]
            
            # Convert back to PIL Image
            output = Image.fromarray(output_array)
            
            # Display the result
            st.image(output, caption="Background Removed")
            
            # Store the result
            st.session_state.analysis_results['background_removed'] = output
            
            # Add download button
            img_byte_arr = BytesIO()
            output.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            st.download_button(
                label="Download Background Removed Image",
                data=img_byte_arr,
                file_name="background_removed.png",
                mime="image/png"
            )

            progress_bar.progress(100)
            status_text.text("Background removal completed!")

        except Exception as e:
            st.error(f"Background removal failed: {str(e)}")
            logger.error(f"Background removal error: {str(e)}", exc_info=True)

    def _process_scene_classification(self, image: Image.Image) -> None:
        """Process scene classification."""
        try:
            st.subheader("Scene Classification")

            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Classifying scene...")
            progress_bar.progress(30)

            # Load and run scene classification model
            from utils import predict_scene
            scene_predictions = predict_scene(image, self.scene_model)

            # Store results
            st.session_state.analysis_results['scene_predictions'] = scene_predictions

            # Display scene classification results
            status_text.text("Displaying results...")
            progress_bar.progress(70)

            st.markdown("### Scene Analysis Results")
            for scene in scene_predictions:
                st.markdown(f"**{scene['scene']}**")
                st.markdown(f"*{scene['description']}*")
                st.markdown(f"""
                <div class="emotion-bar">
                    <div class="emotion-fill" style="width: {scene['probability']}%; background-color: {'#28a745' if scene['probability'] > 50 else '#ffc107'}"></div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"Confidence: {scene['probability']:.2f}%")

            progress_bar.progress(100)
            status_text.text("Scene classification completed!")

        except Exception as e:
            st.error(f"Scene classification failed: {str(e)}")
            logger.error(f"Scene classification error: {str(e)}", exc_info=True)

    def _display_analysis_report(self) -> None:
        """Display the comprehensive analysis report."""
        try:
            st.subheader("Comprehensive Analysis Report")

            if not st.session_state.analysis_results:
                st.warning("No analysis results available. Please process an image first.")
                return

            # Generate report
            from utils import generate_report

            # Get image info
            image_info = {
                "filename": st.session_state.uploaded_file.name,
                "size": st.session_state.uploaded_file.size,
                "type": st.session_state.uploaded_file.type,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            report = generate_report(
                st.session_state.analysis_results.get('face_analysis', []),
                st.session_state.analysis_results.get('scene_predictions', []),
                image_info
            )

            # Display report
            st.markdown("""
            <div class="report-box">
                <h4>Analysis Summary</h4>
            </div>
            """, unsafe_allow_html=True)

            # Display summary information
            st.markdown(f"""
            <div class="report-box">
                <p><strong>Image:</strong> {report['image_info']['filename']}</p>
                <p><strong>Timestamp:</strong> {report['timestamp']}</p>
                <p><strong>Total Faces Detected:</strong> {report['summary']['total_faces']}</p>
                <p><strong>Primary Scene:</strong> {report['summary']['primary_scene']} ({report['summary']['primary_scene_confidence']:.1f}%)</p>
                <p><strong>Dominant Emotion:</strong> {report['summary']['dominant_emotion']['emotion']} ({report['summary']['dominant_emotion']['percentage']:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            # Display face analysis details
            if report['face_analysis']:
                st.markdown("""
                <div class="report-box">
                    <h4>Face Analysis Details</h4>
                </div>
                """, unsafe_allow_html=True)

                for face in report['face_analysis']:
                    with st.expander(f"Face {face['face_id']} Details"):
                        st.markdown(f"**Position:** {face['position']}")
                        st.markdown(f"**Location:** {face['location']['description']}")
                        st.markdown(f"**Size Ratio:** {face['location']['size_ratio']:.1f}%")

                        st.markdown("**Emotions Detected:**")
                        for emotion in face['emotions']:
                            st.markdown(f"- {emotion['emotion']}: {emotion['probability']:.1f}%")

            # Display scene analysis details
            if report['scene_predictions']:
                st.markdown("""
                <div class="report-box">
                    <h4>Scene Analysis Details</h4>
                </div>
                """, unsafe_allow_html=True)

                for scene in report['scene_predictions']:
                    st.markdown(f"""
                    <div class="report-box">
                        <p><strong>{scene['scene']}</strong></p>
                        <p>{scene['description']}</p>
                        <p>Confidence: {scene['probability']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Download report button
            st.download_button(
                label="Download Full Analysis Report",
                data=json.dumps(report, indent=2),
                file_name="analysis_report.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            logger.error(f"Report generation error: {str(e)}", exc_info=True)

def get_available_models() -> list[str]:
    """Get list of available models from the models directory"""
    try:
        if not os.path.exists(MODELS_DIR):
            return []
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
        return models if models else []
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return []

if __name__ == "__main__":
    app = ImageAnalysisApp()
    app.run()
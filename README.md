# 🤖 AnubhavAI - Advanced Image Analysis Platform

AnubhavAI is a sophisticated AI-powered image analysis platform that combines multiple deep learning models to provide comprehensive image analysis capabilities. Built with Python and Streamlit, it offers an intuitive web interface for advanced computer vision tasks.

## 🌟 Key Features

### 1. Face Detection & Analysis
- Multi-angle face detection (both frontal and profile views)
- Advanced face position analysis
- Overlapping face detection handling
- Detailed facial region mapping

### 2. Emotion Recognition
- 7 emotion classifications:
  - Happy (Joy and contentment)
  - Sad (Unhappiness or sorrow)
  - Angry (Displeasure)
  - Surprised (Astonishment)
  - Fearful (Anxiety or worry)
  - Disgusted (Aversion)
  - Neutral (No strong emotion)
- Confidence scores for each emotion
- Detailed emotion descriptions
- Multi-face emotion tracking

### 3. Background Segmentation
- Automatic background removal
- Clean foreground extraction
- Transparent background output
- High-quality edge detection

### 4. Scene Classification
- Environment type detection
- Detailed scene categorization:
  - Indoor/Outdoor classification
  - Natural/Urban environment detection
  - Specific location type identification
- Confidence scores for scene predictions

### 5. Analysis Reporting
- Comprehensive JSON reports
- Detailed image information
- Face detection statistics
- Emotion analysis summaries
- Scene classification results

## 🛠️ Technical Architecture

### Core Components
1. **Frontend**: Streamlit web interface
2. **Backend Services**:
   - Face Detection Engine (OpenCV)
   - Emotion Recognition (PyTorch ResNet50)
   - Background Removal (rembg)
   - Scene Classification (Places365 ResNet)

### Model Stack
- **Face Detection**: Haar Cascade Classifiers
- **Emotion Recognition**: Fine-tuned ResNet50
- **Scene Classification**: Places365 Dataset Model
- **Background Removal**: U2-Net

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Niraj1742/AnubhavAi.git
cd AnubhavAi
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Dependencies
- Python 3.x
- OpenCV (cv2)
- PyTorch & TorchVision
- Streamlit
- NumPy
- Pillow
- rembg
- DeepFace
- TensorFlow
- scikit-learn

## 🎯 Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload an image using the file uploader

4. View the analysis results:
   - Face detection visualization
   - Emotion analysis for each face
   - Background removed version
   - Scene classification results
   - Detailed analysis report

## 💡 Features in Detail

### Face Detection
- Multi-scale face detection
- Profile face detection
- Overlapping face removal
- Position analysis (upper/middle/lower, left/center/right)
- Size ratio calculation

### Emotion Analysis
- Real-time emotion detection
- Multiple face tracking
- Confidence scoring
- Detailed emotion descriptions
- Visual emotion probability bars

### Scene Classification
- Environment type detection
- Location categorization
- Confidence scores
- Detailed scene descriptions

## 🔧 Configuration

The application can be configured through the following parameters:
- Face detection sensitivity
- Emotion confidence thresholds
- Scene classification parameters
- UI customization options



## 🙏 Acknowledgments

- OpenCV team for face detection models
- PyTorch team for deep learning framework
- Places365 dataset team for scene classification
- U2-Net team for background removal
- Streamlit team for the web framework

## 
---
Built with ❤️ by Niraj 

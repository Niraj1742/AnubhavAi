import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import cv2
import numpy as np
from datetime import datetime

def load_face_detector():
    """Load a more accurate face detection model."""
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    return face_cascade, profile_cascade

def detect_faces(image, face_cascade, profile_cascade):
    """Detect faces using multiple cascades for better accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect frontal faces
    frontal_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Detect profile faces
    profile_faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Combine and remove overlapping detections
    all_faces = list(frontal_faces) + list(profile_faces)
    return remove_overlapping_faces(all_faces)

def remove_overlapping_faces(faces, overlap_threshold=0.3):
    """Remove overlapping face detections."""
    if len(faces) == 0:
        return []
    
    # Convert to list of (x, y, w, h) tuples
    faces = [tuple(face) for face in faces]
    
    # Sort by area (largest first)
    faces.sort(key=lambda x: x[2] * x[3], reverse=True)
    
    # Remove overlapping faces
    final_faces = []
    for face in faces:
        overlap = False
        for final_face in final_faces:
            if calculate_overlap(face, final_face) > overlap_threshold:
                overlap = True
                break
        if not overlap:
            final_faces.append(face)
    
    return final_faces

def calculate_overlap(face1, face2):
    """Calculate overlap between two face rectangles."""
    x1 = max(face1[0], face2[0])
    y1 = max(face1[1], face2[1])
    x2 = min(face1[0] + face1[2], face2[0] + face2[2])
    y2 = min(face1[1] + face1[3], face2[1] + face2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = face1[2] * face1[3]
    area2 = face2[2] * face2[3]
    
    return intersection / min(area1, area2)

# ---- Emotion Detection ----
def load_emotion_model():
    """Load the emotion detection model."""
    # Load pre-trained ResNet model for emotion detection
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def get_emotion_labels():
    """Get emotion labels with descriptions."""
    emotions = {
        "Happy": "Positive emotion showing joy and contentment",
        "Sad": "Negative emotion showing unhappiness or sorrow",
        "Angry": "Strong negative emotion showing displeasure",
        "Surprised": "Sudden emotion showing astonishment",
        "Fearful": "Negative emotion showing anxiety or worry",
        "Disgusted": "Strong negative emotion showing aversion",
        "Neutral": "No strong emotion detected"
    }
    return emotions

def detect_emotions(face_img, model):
    """Detect emotions in a face image with detailed analysis."""
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert OpenCV image to PIL
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    
    # Process image
    img_tensor = transform(face_pil).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_emotion = torch.topk(probabilities, 3)
        
    # Get the predicted emotions
    emotions = get_emotion_labels()
    predictions = []
    for prob, emotion_idx in zip(top_prob, top_emotion):
        emotion = list(emotions.keys())[emotion_idx.item() % len(emotions)]
        predictions.append({
            "emotion": emotion,
            "description": emotions[emotion],
            "probability": prob.item() * 100
        })
    
    return predictions

def analyze_face_location(face_img, image_size):
    """Analyze face location in the image."""
    height, width = image_size
    face_height, face_width = face_img.shape[:2]
    
    # Calculate face position relative to image
    position = {
        "top": face_img[0] / height * 100,
        "left": face_img[1] / width * 100,
        "size_ratio": (face_height * face_width) / (height * width) * 100
    }
    
    # Determine face position description
    if position["top"] < 33:
        vertical_pos = "upper"
    elif position["top"] < 66:
        vertical_pos = "middle"
    else:
        vertical_pos = "lower"
        
    if position["left"] < 33:
        horizontal_pos = "left"
    elif position["left"] < 66:
        horizontal_pos = "center"
    else:
        horizontal_pos = "right"
    
    position["description"] = f"{vertical_pos}-{horizontal_pos}"
    return position

# ---- Places365 Scene Classification ----
def load_places_model():
    """Load the pre-trained ResNet model for scene classification."""
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def get_imagenet_labels():
    """Get ImageNet class labels with descriptions."""
    categories = {
        "indoor": "Inside a building or enclosed space",
        "outdoor": "Outside in an open area",
        "nature": "Natural environment like forests or mountains",
        "urban": "City or town environment",
        "building": "Man-made structure",
        "landscape": "Natural scenery",
        "beach": "Coastal area with sand and water",
        "mountain": "Elevated natural formation",
        "forest": "Dense area of trees",
        "city": "Urban area with buildings",
        "street": "Road or pathway",
        "room": "Interior space"
    }
    return categories

def predict_scene(image, model):
    """Predict the scene in the given image with detailed analysis."""
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Process image
    img_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_cat = torch.topk(probabilities, 3)
        
    # Get the predicted categories
    categories = get_imagenet_labels()
    predictions = []
    for prob, cat_idx in zip(top_prob, top_cat):
        category = list(categories.keys())[cat_idx.item() % len(categories)]
        predictions.append({
            "category": category,
            "description": categories[category],
            "probability": prob.item() * 100
        })
    
    return predictions

def generate_report(face_analysis, scene_predictions, image_info):
    """Generate a detailed analysis report."""
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_info": image_info,
        "face_analysis": face_analysis,
        "scene_analysis": scene_predictions,
        "summary": {
            "total_faces": len(face_analysis),
            "primary_scene": scene_predictions[0]["category"] if scene_predictions else "Unknown",
            "dominant_emotion": face_analysis[0]["emotions"][0]["emotion"] if face_analysis else "None"
        }
    }
    return report
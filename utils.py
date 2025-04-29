"""
AnubhavAI - Utility Functions
This module contains utility functions for image analysis including face detection,
emotion recognition, scene classification, and report generation.
"""

# Standard library imports
import os
import json
from datetime import datetime

# Third-party imports
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

def load_face_detector():
    """
    Load face detection models for both frontal and profile face detection.
    
    Returns:
        tuple: (face_cascade, profile_cascade) - OpenCV cascade classifiers
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    return face_cascade, profile_cascade

def detect_faces(image, face_cascade, profile_cascade):
    """
    Detect faces in an image using both frontal and profile face detectors.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        face_cascade: Frontal face detector
        profile_cascade: Profile face detector
    
    Returns:
        list: List of face rectangles (x, y, w, h)
    """
    # Convert image to grayscale for face detection
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
    """
    Remove overlapping face detections to avoid duplicate detections.
    
    Args:
        faces (list): List of face rectangles
        overlap_threshold (float): Threshold for considering faces as overlapping
    
    Returns:
        list: Filtered list of non-overlapping face rectangles
    """
    if len(faces) == 0:
        return []
    
    # Convert to list of tuples
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
    """
    Calculate the overlap ratio between two face rectangles.
    
    Args:
        face1 (tuple): First face rectangle (x, y, w, h)
        face2 (tuple): Second face rectangle (x, y, w, h)
    
    Returns:
        float: Overlap ratio between 0 and 1
    """
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
    """
    Load the pre-trained emotion recognition model.
    
    Returns:
        torch.nn.Module: Loaded emotion recognition model
    """
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def get_emotion_labels():
    """
    Get emotion labels with their descriptions.
    
    Returns:
        dict: Dictionary mapping emotion names to descriptions
    """
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
    """
    Detect emotions in a face image.
    
    Args:
        face_img (numpy.ndarray): Face image in BGR format
        model: Emotion recognition model
    
    Returns:
        list: List of detected emotions with probabilities
    """
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
    """
    Analyze the location of a face within the image.
    
    Args:
        face_img (numpy.ndarray): Face image
        image_size (tuple): Original image size (height, width)
    
    Returns:
        dict: Face location analysis results
    """
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
    """
    Load the pre-trained scene classification model.
    
    Returns:
        torch.nn.Module: Loaded scene classification model
    """
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def get_imagenet_labels():
    """
    Get ImageNet class labels with descriptions.
    
    Returns:
        dict: Dictionary mapping scene categories to descriptions
    """
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
    """
    Predict the scene category of an image.
    
    Args:
        image (PIL.Image): Input image
        model: Scene classification model
    
    Returns:
        list: List of scene predictions with probabilities
    """
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
        top_prob, top_scene = torch.topk(probabilities, 3)
        
    # Get the predicted scenes
    scenes = get_imagenet_labels()
    predictions = []
    for prob, scene_idx in zip(top_prob, top_scene):
        scene = list(scenes.keys())[scene_idx.item() % len(scenes)]
        predictions.append({
            "scene": scene,
            "description": scenes[scene],
            "probability": prob.item() * 100
        })
    
    return predictions

def generate_report(face_analysis, scene_predictions, image_info):
    """
    Generate a comprehensive analysis report.
    
    Args:
        face_analysis (list): List of face analysis results
        scene_predictions (list): List of scene predictions
        image_info (dict): Image metadata
    
    Returns:
        dict: Comprehensive analysis report
    """
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_info": image_info,
        "face_analysis": face_analysis,
        "scene_predictions": scene_predictions,
        "summary": {
            "total_faces": len(face_analysis),
            "primary_scene": scene_predictions[0]["scene"] if scene_predictions else "Unknown",
            "primary_scene_confidence": scene_predictions[0]["probability"] if scene_predictions else 0
        }
    }
    
    return report
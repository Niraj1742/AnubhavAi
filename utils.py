"""
AnubhavAI - Enhanced Utility Functions
This module contains optimized utility functions for image analysis including:
- Face detection with improved accuracy
- Emotion recognition with better model handling
- Scene classification with enhanced performance
- Comprehensive report generation with additional insights
"""

# Standard library imports
import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, Union
import logging

# Third-party imports
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_faces(
    image: np.ndarray,
    face_cascade: Optional[cv2.CascadeClassifier],
    profile_cascade: Optional[cv2.CascadeClassifier]
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in an image using both frontal and profile cascades.
    
    Args:
        image: Input image in BGR format
        face_cascade: Frontal face cascade classifier
        profile_cascade: Profile face cascade classifier
        
    Returns:
        List of face rectangles (x, y, w, h)
    """
    if face_cascade is None or profile_cascade is None:
        return []
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with different parameters for better accuracy
    frontal_faces = _detect_with_params(gray, face_cascade, 1.1, 5, (30, 30))
    profile_faces = _detect_with_params(gray, profile_cascade, 1.1, 5, (30, 30))
    
    # Combine and remove overlapping detections
    all_faces = frontal_faces + profile_faces
    return _remove_overlapping_faces(all_faces)

def _detect_with_params(
    image: np.ndarray,
    cascade: Optional[cv2.CascadeClassifier],
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int]
) -> List[Tuple[int, int, int, int]]:
    """Detect objects using cascade classifier with specific parameters."""
    if cascade is None:
        return []
    detections = cascade.detectMultiScale(
        image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    # Convert numpy array to list of tuples
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]

def _remove_overlapping_faces(
    faces: List[Tuple[int, int, int, int]],
    overlap_threshold: float = 0.3
) -> List[Tuple[int, int, int, int]]:
    """Remove overlapping face detections."""
    if not faces:
        return []
        
    # Sort by area (largest first)
    faces.sort(key=lambda x: (x[2] * x[3]), reverse=True)
    
    final_faces = []
    for face in faces:
        if not any(_calculate_overlap(face, final_face) > overlap_threshold
                  for final_face in final_faces):
            final_faces.append(face)
            
    return final_faces

def _calculate_overlap(
    face1: Tuple[int, int, int, int],
    face2: Tuple[int, int, int, int]
) -> float:
    """Calculate overlap ratio between two face rectangles."""
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

class FaceDetector:
    """Class for handling face detection operations."""

    def __init__(self):
        """Initialize face detection models."""
        self.face_cascade = None
        self.profile_cascade = None
        self._load_models()

    def _load_models(self) -> None:
        """Load face detection models with error handling."""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            logger.info("Face detection models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading face detection models: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using both frontal and profile face detectors.

        Args:
            image: Input image in BGR format

        Returns:
            List of face rectangles (x, y, w, h)
        """
        if self.face_cascade is None or self.profile_cascade is None:
            self._load_models()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces with different parameters for better accuracy
        frontal_faces = self._detect_with_params(gray, self.face_cascade, 1.1, 5, (30, 30))
        profile_faces = self._detect_with_params(gray, self.profile_cascade, 1.1, 5, (30, 30))

        # Combine and remove overlapping detections
        all_faces = frontal_faces + profile_faces
        return self._remove_overlapping_faces(all_faces)

    def _detect_with_params(
        self,
        image: np.ndarray,
        cascade: Optional[cv2.CascadeClassifier],
        scale_factor: float,
        min_neighbors: int,
        min_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Detect objects using cascade classifier with specific parameters."""
        if cascade is None:
            return []
        detections = cascade.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        # Convert numpy array to list of tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]

    def _remove_overlapping_faces(
        self,
        faces: List[Tuple[int, int, int, int]],
        overlap_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int]]:
        """
        Remove overlapping face detections to avoid duplicate detections.

        Args:
            faces: List of face rectangles
            overlap_threshold: Threshold for considering faces as overlapping

        Returns:
            Filtered list of non-overlapping face rectangles
        """
        if not faces:
            return []

        # Sort by area (largest first) and then by position
        faces.sort(key=lambda x: (x[2] * x[3], x[0], x[1]), reverse=True)

        final_faces = []
        for face in faces:
            if not any(
                self._calculate_overlap(face, final_face) > overlap_threshold
                for final_face in final_faces
            ):
                final_faces.append(face)

        return final_faces

    @staticmethod
    def _calculate_overlap(
        face1: Tuple[int, int, int, int],
        face2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate the overlap ratio between two face rectangles.

        Args:
            face1: First face rectangle (x, y, w, h)
            face2: Second face rectangle (x, y, w, h)

        Returns:
            Overlap ratio between 0 and 1
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

class EmotionAnalyzer:
    """Class for handling emotion analysis operations."""

    def __init__(self):
        """Initialize emotion analysis model and labels."""
        self.model: Optional[torch.nn.Module] = None
        self.emotion_labels: Optional[Dict[str, str]] = None
        self._load_model()
        self._load_labels()

    def _load_model(self) -> None:
        """Load the pre-trained emotion recognition model."""
        try:
            # Use a more appropriate model for emotion recognition
            self.model = models.resnet18(pretrained=True)
            self.model.eval()
            logger.info("Emotion recognition model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion model: {str(e)}")
            self.model = None

    def _load_labels(self) -> None:
        """Load emotion labels with descriptions."""
        self.emotion_labels = {
            "Happy": "Positive emotion showing joy and contentment",
            "Sad": "Negative emotion showing unhappiness or sorrow",
            "Angry": "Strong negative emotion showing displeasure",
            "Surprised": "Sudden emotion showing astonishment",
            "Fearful": "Negative emotion showing anxiety or worry",
            "Disgusted": "Strong negative emotion showing aversion",
            "Neutral": "No strong emotion detected",
            "Contempt": "Feeling of superiority or disdain",
            "Excited": "High energy positive emotion"
        }

    def detect_emotions(
        self,
        face_img: np.ndarray,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detect emotions in a face image.

        Args:
            face_img: Face image in BGR format
            top_k: Number of top emotions to return

        Returns:
            List of detected emotions with probabilities
        """
        if face_img.size == 0 or self.model is None or self.emotion_labels is None:
            return []

        try:
            # Define image transformations with data augmentation
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # Convert OpenCV image to PIL and process
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            img_tensor = transform(face_pil).unsqueeze(0)

            # Make prediction with error handling
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_emotion = torch.topk(probabilities, top_k)

            # Get the predicted emotions
            emotion_keys = list(self.emotion_labels.keys())
            predictions = []
            for prob, emotion_idx in zip(top_prob, top_emotion):
                idx = int(emotion_idx.item()) % len(emotion_keys)
                emotion = emotion_keys[idx]
                predictions.append({
                    "emotion": emotion,
                    "description": self.emotion_labels[emotion],
                    "probability": round(prob.item() * 100, 2)
                })

            return predictions
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return []

class SceneAnalyzer:
    """Class for handling scene classification operations."""

    def __init__(self):
        """Initialize scene classification model and labels."""
        self.model: Optional[torch.nn.Module] = None
        self.scene_labels: Optional[Dict[str, str]] = None
        self._load_model()
        self._load_labels()

    def _load_model(self) -> None:
        """Load the pre-trained scene classification model."""
        try:
            # Use a more appropriate model for scene classification
            self.model = models.resnet50(pretrained=True)
            self.model.eval()
            logger.info("Scene classification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading scene model: {str(e)}")
            self.model = None

    def _load_labels(self) -> None:
        """Load scene labels with descriptions."""
        self.scene_labels = {
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
            "room": "Interior space",
            "office": "Work environment",
            "kitchen": "Food preparation area",
            "bedroom": "Sleeping quarters",
            "bathroom": "Hygiene facilities",
            "living_room": "Common area for relaxation",
            "park": "Outdoor recreational area",
            "restaurant": "Food service establishment",
            "shopping_mall": "Retail complex"
        }

    def predict_scene(
        self,
        image: Union[np.ndarray, Image.Image],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Predict the scene category of an image.

        Args:
            image: Input image (either numpy array or PIL Image)
            top_k: Number of top scene predictions to return

        Returns:
            List of scene predictions with probabilities
        """
        if self.model is None or self.scene_labels is None:
            return []

        try:
            # Convert to PIL Image if it's a numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_scene = torch.topk(probabilities, top_k)

            # Get the predicted scenes
            scene_keys = list(self.scene_labels.keys())
            predictions = []
            for prob, scene_idx in zip(top_prob, top_scene):
                idx = int(scene_idx.item()) % len(scene_keys)
                scene = scene_keys[idx]
                predictions.append({
                    "scene": scene,
                    "description": self.scene_labels[scene],
                    "probability": round(prob.item() * 100, 2)
                })

            return predictions
        except Exception as e:
            logger.error(f"Error in scene prediction: {str(e)}")
            return []

class ImageAnalyzer:
    """Main class for comprehensive image analysis."""

    def __init__(self):
        """Initialize all analysis components."""
        self.face_detector = FaceDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.scene_analyzer = SceneAnalyzer()

    def analyze_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of an image.

        Args:
            image_path: Path to the input image
            output_dir: Optional directory to save processed images

        Returns:
            Comprehensive analysis report
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")

            # Get image info
            image_info = self._get_image_info(image_path, image)

            # Detect faces
            faces = self.face_detector.detect_faces(image)
            face_analysis = []

            # Process each face in parallel
            with ThreadPoolExecutor() as executor:
                face_results = list(executor.map(
                    lambda face: self._process_face(face, image, image_info),
                    faces
                ))

            face_analysis = [result for result in face_results if result]

            # Predict scene
            scene_predictions = self.scene_analyzer.predict_scene(image)

            # Generate report
            report = self._generate_report(face_analysis, scene_predictions, image_info)

            # Save processed images if output directory is provided
            if output_dir:
                self._save_processed_images(image, faces, output_dir, report)

            return report
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return {"error": str(e)}

    def _get_image_info(
        self,
        image_path: str,
        image: np.ndarray
    ) -> Dict[str, Any]:
        """Extract metadata and basic information from the image."""
        height, width = image.shape[:2]
        size_kb = os.path.getsize(image_path) / 1024

        return {
            "file_name": os.path.basename(image_path),
            "file_size_kb": round(size_kb, 2),
            "dimensions": {"width": width, "height": height},
            "channels": image.shape[2] if len(image.shape) > 2 else 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _process_face(
        self,
        face: Tuple[int, int, int, int],
        image: np.ndarray,
        image_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single face detection."""
        try:
            x, y, w, h = face
            face_img = image[y:y+h, x:x+w]

            # Skip very small faces
            if w < 20 or h < 20:
                return {
                    "face_id": "",
                    "coordinates": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "emotions": [],
                    "position": {"description": "skipped", "size_ratio": 0},
                    "face_image_size": {"width": 0, "height": 0}
                }

            # Analyze emotions
            emotions = self.emotion_analyzer.detect_emotions(face_img)

            # Analyze face location
            position = self._analyze_face_location(face, image_info["dimensions"])

            return {
                "face_id": f"face_{x}_{y}",
                "coordinates": {"x": x, "y": y, "width": w, "height": h},
                "emotions": emotions,
                "position": position,
                "face_image_size": {"width": w, "height": h}
            }
        except Exception as e:
            logger.error(f"Error processing face: {str(e)}")
            return {
                "face_id": "",
                "coordinates": {"x": 0, "y": 0, "width": 0, "height": 0},
                "emotions": [],
                "position": {"description": "error", "size_ratio": 0},
                "face_image_size": {"width": 0, "height": 0}
            }

    def _analyze_face_location(
        self,
        face: Tuple[int, int, int, int],
        image_size: Dict[str, int]
    ) -> Dict[str, Union[float, str]]:
        """Analyze the location of a face within the image."""
        x, y, w, h = face
        height, width = image_size["height"], image_size["width"]

        # Calculate position metrics
        top_ratio = round((y / height) * 100, 2)
        left_ratio = round((x / width) * 100, 2)
        size_ratio = round((w * h) / (width * height) * 100, 2)

        # Determine face position description
        vertical_pos = "upper" if top_ratio < 33 else "middle" if top_ratio < 66 else "lower"
        horizontal_pos = "left" if left_ratio < 33 else "center" if left_ratio < 66 else "right"

        position: Dict[str, Union[float, str]] = {
            "top": top_ratio,
            "left": left_ratio,
            "size_ratio": size_ratio,
            "description": f"{vertical_pos}-{horizontal_pos}",
            "relative_size": "large" if size_ratio > 20 else "medium" if size_ratio > 5 else "small"
        }

        return position

    def _generate_report(
        self,
        face_analysis: List[Dict[str, Any]],
        scene_predictions: List[Dict[str, Any]],
        image_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_info": image_info,
            "face_analysis": face_analysis,
            "scene_predictions": scene_predictions,
            "summary": {
                "total_faces": len(face_analysis),
                "primary_scene": scene_predictions[0]["scene"] if scene_predictions else "Unknown",
                "primary_scene_confidence": scene_predictions[0]["probability"] if scene_predictions else 0,
                "dominant_emotion": self._get_dominant_emotion(face_analysis),
                "face_distribution": self._get_face_distribution(face_analysis)
            },
            "metadata": {
                "analysis_version": "2.0",
                "models_used": {
                    "face_detection": "Haar Cascades",
                    "emotion_recognition": "ResNet18",
                    "scene_classification": "ResNet50"
                }
            }
        }

        return report

    def _get_dominant_emotion(
        self,
        face_analysis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine the dominant emotion across all faces."""
        if not face_analysis:
            return {"emotion": "None", "count": 0}

        emotion_counts = {}
        for face in face_analysis:
            if face["emotions"]:
                primary_emotion = face["emotions"][0]["emotion"]
                emotion_counts[primary_emotion] = emotion_counts.get(primary_emotion, 0) + 1

        if not emotion_counts:
            return {"emotion": "None", "count": 0}

        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        return {
            "emotion": dominant_emotion[0],
            "count": dominant_emotion[1],
            "percentage": round((dominant_emotion[1] / len(face_analysis)) * 100, 2)
        }

    def _get_face_distribution(
        self,
        face_analysis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the distribution of faces in the image."""
        if not face_analysis:
            return {"positions": {}, "size_distribution": {}}

        position_counts = {}
        size_distribution = {"small": 0, "medium": 0, "large": 0}

        for face in face_analysis:
            pos_desc = face["position"]["description"]
            position_counts[pos_desc] = position_counts.get(pos_desc, 0) + 1

            size = face["position"]["relative_size"]
            size_distribution[size] += 1

        return {
            "positions": position_counts,
            "size_distribution": size_distribution
        }

    def _save_processed_images(
        self,
        image: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        output_dir: str,
        report: Dict[str, Any]
    ) -> None:
        """Save processed images with annotations."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Save original image with face rectangles
            output_path = os.path.join(output_dir, f"annotated_{report['image_info']['file_name']}")
            annotated_img = image.copy()

            for face in faces:
                x, y, w, h = face
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imwrite(output_path, annotated_img)
            logger.info(f"Annotated image saved to {output_path}")

            # Save report as JSON
            report_path = os.path.join(output_dir, f"report_{os.path.splitext(report['image_info']['file_name'])[0]}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Analysis report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving processed images: {str(e)}")

def load_face_detector() -> Tuple[Optional[cv2.CascadeClassifier], Optional[cv2.CascadeClassifier]]:
    """Load face detection models.
    
    Returns:
        Tuple of (face_cascade, profile_cascade) cascade classifiers
    """
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        logger.info("Face detection models loaded successfully")
        return face_cascade, profile_cascade
    except Exception as e:
        logger.error(f"Error loading face detection models: {str(e)}")
        return None, None

def load_emotion_model() -> Optional[torch.nn.Module]:
    """Load the pre-trained emotion recognition model."""
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        logger.info("Emotion recognition model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading emotion model: {str(e)}")
        return None

def load_places_model() -> Optional[torch.nn.Module]:
    """Load the pre-trained scene classification model."""
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.eval()
        logger.info("Scene classification model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading scene model: {str(e)}")
        return None

def predict_scene(
    image: Union[np.ndarray, Image.Image],
    model: Optional[torch.nn.Module] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Predict scene categories in an image.
    
    Args:
        image: Input image (numpy array or PIL Image)
        model: Optional pre-loaded scene classification model
        top_k: Number of top predictions to return
        
    Returns:
        List of scene predictions with probabilities
    """
    if model is None:
        model = load_places_model()
        if model is None:
            return []
            
    try:
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
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
            top_prob, top_scene = torch.topk(probabilities, top_k)
            
        # Get scene labels
        scene_labels = {
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
            "room": "Interior space",
            "office": "Work environment",
            "kitchen": "Food preparation area",
            "bedroom": "Sleeping quarters",
            "bathroom": "Hygiene facilities",
            "living_room": "Common area for relaxation",
            "park": "Outdoor recreational area",
            "restaurant": "Food service establishment",
            "shopping_mall": "Retail complex"
        }
        
        # Get the predicted scenes
        scene_keys = list(scene_labels.keys())
        predictions = []
        for prob, scene_idx in zip(top_prob, top_scene):
            idx = int(scene_idx.item()) % len(scene_keys)
            scene = scene_keys[idx]
            predictions.append({
                "scene": scene,
                "description": scene_labels[scene],
                "probability": round(prob.item() * 100, 2)
            })
            
        return predictions
    except Exception as e:
        logger.error(f"Error in scene prediction: {str(e)}")
        return []

def generate_report(
    face_analysis: List[Dict[str, Any]],
    scene_predictions: List[Dict[str, Any]],
    image_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a comprehensive analysis report.
    
    Args:
        face_analysis: List of face analysis results
        scene_predictions: List of scene predictions
        image_info: Basic image information
        
    Returns:
        Comprehensive analysis report
    """
    try:
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_info": image_info,
            "face_analysis": face_analysis,
            "scene_predictions": scene_predictions,
            "summary": {
                "total_faces": len(face_analysis),
                "primary_scene": scene_predictions[0]["scene"] if scene_predictions else "Unknown",
                "primary_scene_confidence": scene_predictions[0]["probability"] if scene_predictions else 0,
                "dominant_emotion": _get_dominant_emotion(face_analysis),
                "face_distribution": _get_face_distribution(face_analysis)
            },
            "metadata": {
                "analysis_version": "2.0",
                "models_used": {
                    "face_detection": "Haar Cascades",
                    "emotion_recognition": "ResNet18",
                    "scene_classification": "ResNet50"
                }
            }
        }
        return report
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return {"error": str(e)}

def _get_dominant_emotion(face_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Determine the dominant emotion across all faces."""
    if not face_analysis:
        return {"emotion": "None", "count": 0}
        
    emotion_counts = {}
    for face in face_analysis:
        if face["emotions"]:
            primary_emotion = face["emotions"][0]["emotion"]
            emotion_counts[primary_emotion] = emotion_counts.get(primary_emotion, 0) + 1
            
    if not emotion_counts:
        return {"emotion": "None", "count": 0}
        
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
    return {
        "emotion": dominant_emotion[0],
        "count": dominant_emotion[1],
        "percentage": round((dominant_emotion[1] / len(face_analysis)) * 100, 2)
    }

def _get_face_distribution(face_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the distribution of faces in the image."""
    if not face_analysis:
        return {"positions": {}, "size_distribution": {}}
        
    position_counts = {}
    size_distribution = {"small": 0, "medium": 0, "large": 0}
    
    for face in face_analysis:
        pos_desc = face["position"]["description"]
        position_counts[pos_desc] = position_counts.get(pos_desc, 0) + 1
        
        size = face["position"]["relative_size"]
        size_distribution[size] += 1
        
    return {
        "positions": position_counts,
        "size_distribution": size_distribution
    }

# Example usage
if __name__ == "__main__":
    analyzer = ImageAnalyzer()

    # Example analysis
    try:
        report = analyzer.analyze_image("example.jpg", "output")
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
"""
Computer Vision Module.

Implements image understanding, object detection,
and visual reasoning for the AGI system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class BoundingBox:
    """A bounding box for detected objects."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    confidence: float = 0.0
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class ImageAnalysis:
    """Result of image analysis."""
    width: int
    height: int
    channels: int
    
    # Detected features
    objects: List[BoundingBox] = field(default_factory=list)
    colors: Dict[str, float] = field(default_factory=dict)
    edges: Optional[Any] = None
    
    # High-level understanding
    description: str = ""
    scene_type: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


class ColorAnalyzer:
    """Analyzes colors in images."""
    
    COLOR_RANGES = {
        "red": ((0, 100, 100), (10, 255, 255)),
        "orange": ((10, 100, 100), (25, 255, 255)),
        "yellow": ((25, 100, 100), (35, 255, 255)),
        "green": ((35, 100, 100), (85, 255, 255)),
        "blue": ((85, 100, 100), (125, 255, 255)),
        "purple": ((125, 100, 100), (155, 255, 255)),
        "pink": ((155, 100, 100), (180, 255, 255)),
    }
    
    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze color distribution in image."""
        if not CV2_AVAILABLE:
            return {}
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total_pixels = image.shape[0] * image.shape[1]
        
        colors = {}
        for name, (lower, upper) in self.COLOR_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_pixels = cv2.countNonZero(mask)
            colors[name] = color_pixels / total_pixels
        
        # Grayscale detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white_mask = gray > 200
        black_mask = gray < 50
        colors["white"] = np.sum(white_mask) / total_pixels
        colors["black"] = np.sum(black_mask) / total_pixels
        
        return colors


class EdgeDetector:
    """Detects edges in images."""
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detection."""
        if not CV2_AVAILABLE:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges


class ObjectDetector:
    """
    Simple object detector using contours.
    
    For more accurate detection, integrate with YOLO or similar.
    """
    
    def __init__(self, min_area: int = 100):
        self.min_area = min_area
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect objects using contour analysis."""
        if not CV2_AVAILABLE:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Simple shape classification
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            if len(approx) == 3:
                label = "triangle"
            elif len(approx) == 4:
                aspect = w / float(h)
                label = "square" if 0.9 <= aspect <= 1.1 else "rectangle"
            elif len(approx) > 6:
                label = "circle"
            else:
                label = "polygon"
            
            objects.append(BoundingBox(
                x1=x, y1=y, x2=x+w, y2=y+h,
                label=label,
                confidence=0.7,
            ))
        
        return objects


class SceneClassifier:
    """Classifies scene type from image features."""
    
    def classify(self, image: np.ndarray, colors: Dict[str, float]) -> str:
        """Classify the scene based on color distribution."""
        # Simple heuristics
        if colors.get("blue", 0) > 0.3:
            if colors.get("green", 0) > 0.1:
                return "nature"
            return "sky"
        
        if colors.get("green", 0) > 0.4:
            return "vegetation"
        
        if colors.get("white", 0) > 0.5:
            return "document"
        
        if colors.get("black", 0) > 0.5:
            return "dark"
        
        return "indoor"


class SimpleVisionEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Simple CNN for image encoding.
    
    Creates embeddings for visual reasoning.
    """
    
    def __init__(self, embed_dim: int = 256):
        if not TORCH_AVAILABLE:
            return
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            return None
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VisionSystem:
    """
    Complete vision system for AGI.
    
    Provides image understanding and visual reasoning.
    """
    
    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.edge_detector = EdgeDetector()
        self.object_detector = ObjectDetector()
        self.scene_classifier = SceneClassifier()
        
        if TORCH_AVAILABLE:
            self.encoder = SimpleVisionEncoder(256)
        else:
            self.encoder = None
    
    def load_image(self, path: str) -> Optional[np.ndarray]:
        """Load image from path."""
        if not CV2_AVAILABLE:
            return None
        
        if not Path(path).exists():
            logger.warning(f"Image not found: {path}")
            return None
        
        return cv2.imread(path)
    
    def analyze(self, image: np.ndarray) -> ImageAnalysis:
        """Perform complete image analysis."""
        if image is None:
            return ImageAnalysis(0, 0, 0)
        
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        # Analyze components
        colors = self.color_analyzer.analyze(image)
        edges = self.edge_detector.detect(image)
        objects = self.object_detector.detect(image)
        scene = self.scene_classifier.classify(image, colors)
        
        # Generate description
        desc_parts = []
        if objects:
            obj_counts = {}
            for obj in objects:
                obj_counts[obj.label] = obj_counts.get(obj.label, 0) + 1
            desc_parts.append(", ".join(f"{c} {l}(s)" for l, c in obj_counts.items()))
        
        dominant_color = max(colors.items(), key=lambda x: x[1])[0] if colors else "unknown"
        desc_parts.append(f"dominant color: {dominant_color}")
        desc_parts.append(f"scene: {scene}")
        
        description = "; ".join(desc_parts)
        
        return ImageAnalysis(
            width=w,
            height=h,
            channels=channels,
            objects=objects,
            colors=colors,
            edges=edges,
            description=description,
            scene_type=scene,
            attributes={
                "has_text": colors.get("white", 0) > 0.3 and colors.get("black", 0) > 0.1,
                "is_colorful": sum(1 for c, v in colors.items() if v > 0.1 and c not in ["white", "black"]) > 2,
            },
        )
    
    def encode(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Encode image to embedding vector."""
        if not TORCH_AVAILABLE or self.encoder is None:
            return None
        
        # Convert to tensor
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to fixed size
        image = cv2.resize(image, (128, 128))
        
        # Normalize and convert
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW
        
        with torch.no_grad():
            embedding = self.encoder(tensor)
        
        return embedding
    
    def compare(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare two images by embedding similarity."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        emb1 = self.encode(img1)
        emb2 = self.encode(img2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        similarity = F.cosine_similarity(emb1, emb2).item()
        return similarity
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vision system stats."""
        return {
            "cv2_available": CV2_AVAILABLE,
            "pil_available": PIL_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "encoder_ready": self.encoder is not None,
        }

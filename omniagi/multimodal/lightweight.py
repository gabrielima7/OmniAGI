"""
Lightweight Multi-Modal Processor.

Uses sentence-transformers for embeddings (works on CPU/GPU).
Supports text, images, and basic vision.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class MultiModalInput:
    """A multi-modal input."""
    
    text: str = ""
    image_path: str = ""
    image_data: bytes = None
    audio_path: str = ""
    modality: str = "text"


@dataclass
class MultiModalOutput:
    """Output from multi-modal processing."""
    
    text: str
    embedding: list[float] = None
    image_description: str = ""
    confidence: float = 0.0


class LightweightMultiModal:
    """
    Lightweight Multi-Modal Processor.
    
    Uses:
    - sentence-transformers for text/image embeddings
    - PIL for image processing
    - No heavy models (works on 6GB GPU)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = None
        self._model_name = model_name
        self._initialized = False
        
        logger.info("Lightweight MultiModal initializing", model=model_name)
    
    def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(self._model_name)
            self._initialized = True
            
            logger.info("MultiModal initialized", model=self._model_name)
            return True
            
        except Exception as e:
            logger.error("MultiModal init failed", error=str(e))
            return False
    
    def process_text(self, text: str) -> MultiModalOutput:
        """Process text input and create embedding."""
        if not self._initialized:
            self.initialize()
        
        embedding = self._model.encode(text).tolist()
        
        return MultiModalOutput(
            text=text,
            embedding=embedding,
            confidence=1.0,
        )
    
    def process_image(self, image_path: str) -> MultiModalOutput:
        """Process image and create description + embedding."""
        try:
            from PIL import Image
            
            img = Image.open(image_path)
            
            # Get basic image info
            width, height = img.size
            mode = img.mode
            
            # Create text description
            description = f"Image: {width}x{height} {mode} format"
            
            # Get dominant colors
            if mode == "RGB":
                img_small = img.resize((50, 50))
                pixels = list(img_small.getdata())
                avg_r = sum(p[0] for p in pixels) // len(pixels)
                avg_g = sum(p[1] for p in pixels) // len(pixels)
                avg_b = sum(p[2] for p in pixels) // len(pixels)
                description += f", dominant color RGB({avg_r},{avg_g},{avg_b})"
            
            # Create embedding from description
            if self._initialized:
                embedding = self._model.encode(description).tolist()
            else:
                embedding = []
            
            return MultiModalOutput(
                text=description,
                embedding=embedding,
                image_description=description,
                confidence=0.8,
            )
            
        except Exception as e:
            logger.error("Image processing failed", error=str(e))
            return MultiModalOutput(
                text=f"Error: {e}",
                confidence=0.0,
            )
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not self._initialized:
            self.initialize()
        
        from sentence_transformers import util
        
        emb1 = self._model.encode(text1)
        emb2 = self._model.encode(text2)
        
        similarity = util.cos_sim(emb1, emb2).item()
        return similarity
    
    def is_initialized(self) -> bool:
        return self._initialized

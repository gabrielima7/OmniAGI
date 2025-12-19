"""
Sensory Processor - Unified sensory processing.

Processes inputs from multiple simulated senses
into a unified representation for the AI.
"""

from __future__ import annotations

import json
import logging

try:
    import structlog
except ImportError:
    structlog = None
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors."""
    
    VISION = auto()       # Visual perception
    PROPRIOCEPTION = auto()  # Body awareness
    TOUCH = auto()        # Tactile sensing
    AUDIO = auto()        # Auditory perception
    PROXIMITY = auto()    # Distance sensing
    FORCE = auto()        # Force/torque sensing


@dataclass
class SensorReading:
    """A single sensor reading."""
    
    sensor_type: SensorType
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "type": self.sensor_type.name,
            "timestamp": self.timestamp,
            "data": self.data,
            "confidence": self.confidence,
        }


@dataclass
class ProprioceptiveState:
    """Body state awareness."""
    
    position: tuple[float, float, float] = (0, 0, 0)
    rotation: tuple[float, float, float] = (0, 0, 0)
    velocity: tuple[float, float, float] = (0, 0, 0)
    
    # Joint states (if applicable)
    joint_positions: dict[str, float] = field(default_factory=dict)
    joint_velocities: dict[str, float] = field(default_factory=dict)
    
    # Grasp state
    grasping: bool = False
    grasped_object: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "rotation": self.rotation,
            "velocity": self.velocity,
            "joints": self.joint_positions,
            "grasping": self.grasping,
            "grasped_object": self.grasped_object,
        }


@dataclass
class VisualPercept:
    """Visual perception data."""
    
    objects_visible: list[dict] = field(default_factory=list)
    depth_map: list[list[float]] | None = None
    field_of_view: float = 90.0  # Degrees
    
    def to_dict(self) -> dict:
        return {
            "objects": self.objects_visible,
            "has_depth": self.depth_map is not None,
            "fov": self.field_of_view,
        }


class SensoryProcessor:
    """
    Unified sensory processing system.
    
    Integrates multiple sensor modalities into
    a coherent representation for decision-making.
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Current sensor readings
        self._readings: dict[SensorType, SensorReading] = {}
        
        # Sensor buffers for temporal processing
        self._buffers: dict[SensorType, list[SensorReading]] = {
            sensor: [] for sensor in SensorType
        }
        self._buffer_size = 30
        
        # Internal state
        self._proprioception = ProprioceptiveState()
        self._visual = VisualPercept()
        
        # Attention weights
        self._attention: dict[SensorType, float] = {
            SensorType.VISION: 1.0,
            SensorType.PROPRIOCEPTION: 0.8,
            SensorType.TOUCH: 0.7,
            SensorType.PROXIMITY: 0.6,
            SensorType.AUDIO: 0.5,
            SensorType.FORCE: 0.5,
        }
        
        logger.info("Sensory Processor initialized")
    
    def update_reading(self, reading: SensorReading) -> None:
        """Update with a new sensor reading."""
        self._readings[reading.sensor_type] = reading
        
        # Add to buffer
        buffer = self._buffers[reading.sensor_type]
        buffer.append(reading)
        if len(buffer) > self._buffer_size:
            buffer.pop(0)
        
        # Update internal state based on reading type
        if reading.sensor_type == SensorType.PROPRIOCEPTION:
            self._update_proprioception(reading)
        elif reading.sensor_type == SensorType.VISION:
            self._update_vision(reading)
    
    def _update_proprioception(self, reading: SensorReading) -> None:
        """Update proprioceptive state."""
        data = reading.data
        
        if "position" in data:
            self._proprioception.position = tuple(data["position"])
        if "rotation" in data:
            self._proprioception.rotation = tuple(data["rotation"])
        if "velocity" in data:
            self._proprioception.velocity = tuple(data["velocity"])
        if "grasping" in data:
            self._proprioception.grasping = data["grasping"]
            self._proprioception.grasped_object = data.get("grasped_object")
    
    def _update_vision(self, reading: SensorReading) -> None:
        """Update visual percept."""
        data = reading.data
        
        if "objects" in data:
            self._visual.objects_visible = data["objects"]
        if "depth_map" in data:
            self._visual.depth_map = data["depth_map"]
    
    def from_simulation(self, sim_state: dict) -> None:
        """Update sensors from simulation state."""
        timestamp = sim_state.get("time", 0.0)
        
        # Proprioception from agent state
        if "agent" in sim_state:
            agent = sim_state["agent"]
            self.update_reading(SensorReading(
                sensor_type=SensorType.PROPRIOCEPTION,
                timestamp=timestamp,
                data={
                    "position": list(agent.get("position", {}).values()),
                    "rotation": list(agent.get("rotation", {}).values()),
                },
            ))
        
        # Vision from objects
        if "objects" in sim_state:
            visible_objects = [
                {
                    "id": obj_id,
                    "name": obj.get("name", ""),
                    "position": list(obj.get("position", {}).values()),
                    "type": obj.get("type", "unknown"),
                }
                for obj_id, obj in sim_state["objects"].items()
            ]
            
            self.update_reading(SensorReading(
                sensor_type=SensorType.VISION,
                timestamp=timestamp,
                data={"objects": visible_objects},
            ))
    
    def get_reading(self, sensor_type: SensorType) -> SensorReading | None:
        """Get latest reading from a sensor."""
        return self._readings.get(sensor_type)
    
    def get_proprioception(self) -> ProprioceptiveState:
        """Get current proprioceptive state."""
        return self._proprioception
    
    def get_visual(self) -> VisualPercept:
        """Get current visual percept."""
        return self._visual
    
    def get_unified_percept(self) -> dict[str, Any]:
        """
        Get unified sensory percept.
        
        Combines all sensor modalities weighted by attention.
        """
        percept = {
            "timestamp": datetime.now().timestamp(),
            "body": self._proprioception.to_dict(),
            "vision": self._visual.to_dict(),
            "readings": {},
            "attention": dict(self._attention),
        }
        
        for sensor_type, reading in self._readings.items():
            weight = self._attention.get(sensor_type, 0.5)
            if weight > 0.3:  # Only include attended sensors
                percept["readings"][sensor_type.name] = {
                    "data": reading.data,
                    "confidence": reading.confidence * weight,
                }
        
        return percept
    
    def set_attention(self, sensor_type: SensorType, weight: float) -> None:
        """Set attention weight for a sensor type."""
        self._attention[sensor_type] = max(0.0, min(1.0, weight))
    
    def detect_change(self, sensor_type: SensorType) -> bool:
        """Detect if there was a significant change in sensor."""
        buffer = self._buffers[sensor_type]
        if len(buffer) < 2:
            return False
        
        # Simple change detection
        latest = buffer[-1]
        prev = buffer[-2]
        
        return latest.data != prev.data
    
    def get_motion(self) -> tuple[float, float, float]:
        """Estimate current motion from proprioception."""
        return self._proprioception.velocity
    
    def is_colliding(self) -> bool:
        """Check if in collision (from touch/force sensors)."""
        touch = self._readings.get(SensorType.TOUCH)
        if touch and touch.data.get("contact", False):
            return True
        
        force = self._readings.get(SensorType.FORCE)
        if force and force.data.get("magnitude", 0) > 10.0:
            return True
        
        return False
    
    def get_nearest_object(self) -> dict | None:
        """Get nearest visible object."""
        if not self._visual.objects_visible:
            return None
        
        agent_pos = self._proprioception.position
        
        nearest = None
        min_dist = float("inf")
        
        for obj in self._visual.objects_visible:
            obj_pos = obj.get("position", [0, 0, 0])
            dist = sum((a - b) ** 2 for a, b in zip(agent_pos, obj_pos)) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest = obj
        
        return nearest
    
    def to_dict(self) -> dict:
        """Serialize current state."""
        return {
            "proprioception": self._proprioception.to_dict(),
            "vision": self._visual.to_dict(),
            "readings": {
                k.name: v.to_dict()
                for k, v in self._readings.items()
            },
        }

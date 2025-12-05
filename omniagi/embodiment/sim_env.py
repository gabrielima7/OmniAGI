"""
Simulation Environment - Physical world simulation.

Provides interfaces to physics simulators for embodied
AI grounding and learning through interaction.
"""

from __future__ import annotations

import json
import structlog
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

logger = structlog.get_logger()


class PhysicsBackend(Enum):
    """Supported physics backends."""
    
    SIMPLE = auto()      # Built-in simple physics
    PYBULLET = auto()    # PyBullet simulator
    MUJOCO = auto()      # MuJoCo simulator
    HABITAT = auto()     # Habitat for navigation


@dataclass
class Vector3:
    """3D vector for positions and velocities."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class PhysicalObject:
    """A physical object in the simulation."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    object_type: str = "generic"
    
    # Transform
    position: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    
    # Physical properties
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.3  # Bounciness
    
    # State
    is_static: bool = False
    is_graspable: bool = True
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.object_type,
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "velocity": self.velocity.to_dict(),
            "mass": self.mass,
            "is_static": self.is_static,
        }


@dataclass
class SimulationState:
    """Current state of the simulation."""
    
    time: float = 0.0
    step: int = 0
    objects: dict[str, PhysicalObject] = field(default_factory=dict)
    agent_position: Vector3 = field(default_factory=Vector3)
    agent_rotation: Vector3 = field(default_factory=Vector3)
    
    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "step": self.step,
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "agent": {
                "position": self.agent_position.to_dict(),
                "rotation": self.agent_rotation.to_dict(),
            },
        }


class PhysicsEngine(ABC):
    """Abstract physics engine interface."""
    
    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds."""
        pass
    
    @abstractmethod
    def add_object(self, obj: PhysicalObject) -> str:
        """Add object to simulation."""
        pass
    
    @abstractmethod
    def remove_object(self, obj_id: str) -> bool:
        """Remove object from simulation."""
        pass
    
    @abstractmethod
    def apply_force(self, obj_id: str, force: Vector3) -> bool:
        """Apply force to object."""
        pass
    
    @abstractmethod
    def get_state(self) -> SimulationState:
        """Get current simulation state."""
        pass


class SimplePhysicsEngine(PhysicsEngine):
    """
    Simple built-in physics engine.
    
    Basic physics simulation for testing and simple scenarios.
    """
    
    def __init__(self, gravity: float = -9.81):
        self.gravity = gravity
        self._objects: dict[str, PhysicalObject] = {}
        self._time = 0.0
        self._step = 0
        self._agent_position = Vector3(0, 0, 0)
        self._agent_rotation = Vector3(0, 0, 0)
        
        logger.info("Simple physics engine initialized")
    
    def step(self, dt: float = 0.016) -> None:
        """Advance simulation."""
        for obj in self._objects.values():
            if obj.is_static:
                continue
            
            # Apply gravity
            obj.velocity.y += self.gravity * dt
            
            # Apply velocity
            obj.position = obj.position + obj.velocity * dt
            
            # Simple ground collision
            if obj.position.y < 0:
                obj.position.y = 0
                obj.velocity.y = -obj.velocity.y * obj.restitution
        
        self._time += dt
        self._step += 1
    
    def add_object(self, obj: PhysicalObject) -> str:
        """Add object to simulation."""
        self._objects[obj.id] = obj
        logger.debug("Object added", id=obj.id, name=obj.name)
        return obj.id
    
    def remove_object(self, obj_id: str) -> bool:
        """Remove object from simulation."""
        if obj_id in self._objects:
            del self._objects[obj_id]
            return True
        return False
    
    def apply_force(self, obj_id: str, force: Vector3) -> bool:
        """Apply force to object."""
        if obj_id not in self._objects:
            return False
        
        obj = self._objects[obj_id]
        if obj.is_static:
            return False
        
        # F = ma, so a = F/m
        acceleration = force * (1.0 / obj.mass)
        obj.velocity = obj.velocity + acceleration
        return True
    
    def get_state(self) -> SimulationState:
        """Get current simulation state."""
        return SimulationState(
            time=self._time,
            step=self._step,
            objects=self._objects.copy(),
            agent_position=self._agent_position,
            agent_rotation=self._agent_rotation,
        )
    
    def move_agent(self, direction: Vector3, speed: float = 1.0) -> None:
        """Move the agent."""
        self._agent_position = self._agent_position + direction * speed
    
    def rotate_agent(self, rotation: Vector3) -> None:
        """Rotate the agent."""
        self._agent_rotation = self._agent_rotation + rotation


class SimulationEnvironment:
    """
    High-level simulation environment.
    
    Manages the physics engine and provides
    a clean interface for AI interaction.
    """
    
    def __init__(
        self,
        backend: PhysicsBackend = PhysicsBackend.SIMPLE,
        storage_path: Path | str | None = None,
    ):
        self.backend = backend
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Initialize physics engine
        self._engine = self._create_engine(backend)
        
        # State tracking
        self._history: list[dict] = []
        self._max_history = 1000
        
        logger.info("Simulation Environment initialized", backend=backend.name)
    
    def _create_engine(self, backend: PhysicsBackend) -> PhysicsEngine:
        """Create physics engine based on backend."""
        if backend == PhysicsBackend.SIMPLE:
            return SimplePhysicsEngine()
        else:
            # For other backends, try to import them
            # Fall back to simple if not available
            logger.warning(
                "Backend not available, using simple physics",
                requested=backend.name,
            )
            return SimplePhysicsEngine()
    
    def step(self, dt: float = 0.016) -> SimulationState:
        """Advance simulation by one step."""
        self._engine.step(dt)
        state = self._engine.get_state()
        
        # Record history
        self._history.append(state.to_dict())
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        return state
    
    def spawn_object(
        self,
        name: str,
        position: tuple[float, float, float] = (0, 0, 0),
        object_type: str = "box",
        mass: float = 1.0,
        is_static: bool = False,
    ) -> str:
        """Spawn an object in the environment."""
        obj = PhysicalObject(
            name=name,
            object_type=object_type,
            position=Vector3(*position),
            mass=mass,
            is_static=is_static,
        )
        return self._engine.add_object(obj)
    
    def remove_object(self, obj_id: str) -> bool:
        """Remove an object."""
        return self._engine.remove_object(obj_id)
    
    def push_object(
        self,
        obj_id: str,
        direction: tuple[float, float, float],
        force: float = 10.0,
    ) -> bool:
        """Push an object with force."""
        force_vec = Vector3(*direction) * force
        return self._engine.apply_force(obj_id, force_vec)
    
    def move_agent(
        self,
        direction: tuple[float, float, float],
        speed: float = 1.0,
    ) -> None:
        """Move the agent."""
        if isinstance(self._engine, SimplePhysicsEngine):
            self._engine.move_agent(Vector3(*direction), speed)
    
    def get_state(self) -> SimulationState:
        """Get current state."""
        return self._engine.get_state()
    
    def get_objects_in_range(
        self,
        position: tuple[float, float, float],
        radius: float,
    ) -> list[PhysicalObject]:
        """Get objects within radius of position."""
        state = self._engine.get_state()
        pos = Vector3(*position)
        
        in_range = []
        for obj in state.objects.values():
            dx = obj.position.x - pos.x
            dy = obj.position.y - pos.y
            dz = obj.position.z - pos.z
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            if dist <= radius:
                in_range.append(obj)
        
        return in_range
    
    def raycast(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_distance: float = 100.0,
    ) -> tuple[PhysicalObject | None, float]:
        """
        Cast a ray and find first hit.
        
        Returns (hit_object, distance) or (None, max_distance).
        """
        # Simplified raycast - just checks objects along ray
        state = self._engine.get_state()
        
        closest_obj = None
        closest_dist = max_distance
        
        for obj in state.objects.values():
            # Simple distance check (proper raycast would check intersection)
            dx = obj.position.x - origin[0]
            dy = obj.position.y - origin[1]
            dz = obj.position.z - origin[2]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            if dist < closest_dist:
                closest_dist = dist
                closest_obj = obj
        
        return closest_obj, closest_dist
    
    def reset(self) -> SimulationState:
        """Reset the environment."""
        self._engine = self._create_engine(self.backend)
        self._history.clear()
        return self._engine.get_state()
    
    def get_history(self, n: int = 10) -> list[dict]:
        """Get recent history."""
        return self._history[-n:]
    
    def save_state(self) -> None:
        """Save current state to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "state": self._engine.get_state().to_dict(),
                "history": self._history[-100:],
            }, f, indent=2)

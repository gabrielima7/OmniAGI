"""
Embodiment Module.

Implements action-perception loop and physical world
simulation interface for embodied AGI.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of embodied actions."""
    MOVE = auto()
    GRASP = auto()
    RELEASE = auto()
    PUSH = auto()
    PULL = auto()
    ROTATE = auto()
    LOOK = auto()
    SPEAK = auto()


@dataclass
class Vector3:
    """3D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> "Vector3":
        mag = self.magnitude()
        if mag > 0:
            return Vector3(self.x/mag, self.y/mag, self.z/mag)
        return Vector3()
    
    def distance_to(self, other: "Vector3") -> float:
        return (self - other).magnitude()


@dataclass
class WorldObject:
    """An object in the simulated world."""
    id: str
    name: str
    position: Vector3 = field(default_factory=Vector3)
    size: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    
    # Physical properties
    mass: float = 1.0
    is_static: bool = False
    is_graspable: bool = True
    
    # State
    velocity: Vector3 = field(default_factory=Vector3)
    is_grasped: bool = False


@dataclass
class Perception:
    """Sensory perception data."""
    visible_objects: List[WorldObject] = field(default_factory=list)
    tactile_feedback: Dict[str, float] = field(default_factory=dict)
    proprioception: Dict[str, float] = field(default_factory=dict)
    audio: List[str] = field(default_factory=list)


@dataclass
class Action:
    """An action to execute in the world."""
    type: ActionType
    target: Optional[str] = None  # Object ID
    direction: Optional[Vector3] = None
    magnitude: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    action: Action
    perception_after: Perception
    feedback: str = ""
    energy_cost: float = 0.0


class SimplePhysicsEngine:
    """
    Simple physics simulation.
    
    Handles basic physics without external dependencies.
    """
    
    def __init__(self, gravity: float = -9.8, dt: float = 0.016):
        self.gravity = gravity
        self.dt = dt
        self.objects: Dict[str, WorldObject] = {}
        self.ground_level = 0.0
    
    def add_object(self, obj: WorldObject) -> None:
        """Add an object to the simulation."""
        self.objects[obj.id] = obj
    
    def step(self) -> None:
        """Advance physics by one timestep."""
        for obj in self.objects.values():
            if obj.is_static or obj.is_grasped:
                continue
            
            # Apply gravity
            obj.velocity.y += self.gravity * self.dt
            
            # Update position
            obj.position = obj.position + obj.velocity * self.dt
            
            # Ground collision
            if obj.position.y < self.ground_level:
                obj.position.y = self.ground_level
                obj.velocity.y = -obj.velocity.y * 0.3  # Bounce
                obj.velocity = obj.velocity * 0.8  # Friction
    
    def apply_force(self, obj_id: str, force: Vector3) -> bool:
        """Apply a force to an object."""
        if obj_id not in self.objects:
            return False
        
        obj = self.objects[obj_id]
        if obj.is_static:
            return False
        
        acceleration = force * (1.0 / obj.mass)
        obj.velocity = obj.velocity + acceleration * self.dt
        return True


class EmbodiedAgent:
    """
    Embodied agent with sensorimotor capabilities.
    
    Implements action-perception loop for physical world interaction.
    """
    
    def __init__(self):
        self.position = Vector3(0, 0, 0)
        self.orientation = 0.0  # Facing direction in radians
        
        self.grasped_object: Optional[str] = None
        self.reach_distance = 2.0
        self.move_speed = 1.0
        
        self.energy = 100.0
        self.max_energy = 100.0
        
        self.physics = SimplePhysicsEngine()
        self.perception_history: List[Perception] = []
    
    def perceive(self) -> Perception:
        """Get current sensory perception."""
        visible = []
        
        for obj in self.physics.objects.values():
            distance = self.position.distance_to(obj.position)
            if distance < 10.0:  # Vision range
                visible.append(obj)
        
        # Sort by distance
        visible.sort(key=lambda o: self.position.distance_to(o.position))
        
        perception = Perception(
            visible_objects=visible,
            tactile_feedback={"grasping": 1.0 if self.grasped_object else 0.0},
            proprioception={
                "position_x": self.position.x,
                "position_y": self.position.y,
                "position_z": self.position.z,
                "orientation": self.orientation,
                "energy": self.energy,
            },
        )
        
        self.perception_history.append(perception)
        return perception
    
    def execute(self, action: Action) -> ActionResult:
        """Execute an action in the world."""
        energy_cost = 1.0
        success = False
        feedback = ""
        
        if action.type == ActionType.MOVE:
            success, feedback = self._execute_move(action)
            energy_cost = 2.0
        
        elif action.type == ActionType.GRASP:
            success, feedback = self._execute_grasp(action)
            energy_cost = 1.5
        
        elif action.type == ActionType.RELEASE:
            success, feedback = self._execute_release(action)
            energy_cost = 0.5
        
        elif action.type == ActionType.PUSH:
            success, feedback = self._execute_push(action)
            energy_cost = 3.0
        
        elif action.type == ActionType.LOOK:
            success, feedback = self._execute_look(action)
            energy_cost = 0.1
        
        # Update physics
        self.physics.step()
        
        # Consume energy
        self.energy = max(0, self.energy - energy_cost)
        
        return ActionResult(
            success=success,
            action=action,
            perception_after=self.perceive(),
            feedback=feedback,
            energy_cost=energy_cost,
        )
    
    def _execute_move(self, action: Action) -> Tuple[bool, str]:
        """Execute a movement action."""
        if action.direction is None:
            return False, "No direction specified"
        
        move_vec = action.direction.normalize() * self.move_speed * action.magnitude
        self.position = self.position + move_vec
        
        # Also move grasped object
        if self.grasped_object and self.grasped_object in self.physics.objects:
            obj = self.physics.objects[self.grasped_object]
            obj.position = obj.position + move_vec
        
        return True, f"Moved to {self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f}"
    
    def _execute_grasp(self, action: Action) -> Tuple[bool, str]:
        """Execute a grasping action."""
        if self.grasped_object:
            return False, "Already holding an object"
        
        if action.target is None:
            return False, "No target specified"
        
        if action.target not in self.physics.objects:
            return False, "Object not found"
        
        obj = self.physics.objects[action.target]
        distance = self.position.distance_to(obj.position)
        
        if distance > self.reach_distance:
            return False, f"Object too far ({distance:.1f} > {self.reach_distance})"
        
        if not obj.is_graspable:
            return False, "Object cannot be grasped"
        
        self.grasped_object = action.target
        obj.is_grasped = True
        obj.velocity = Vector3()
        
        return True, f"Grasped {obj.name}"
    
    def _execute_release(self, action: Action) -> Tuple[bool, str]:
        """Execute a release action."""
        if not self.grasped_object:
            return False, "Not holding anything"
        
        obj = self.physics.objects[self.grasped_object]
        obj.is_grasped = False
        
        released_name = obj.name
        self.grasped_object = None
        
        return True, f"Released {released_name}"
    
    def _execute_push(self, action: Action) -> Tuple[bool, str]:
        """Execute a pushing action."""
        if action.target is None or action.direction is None:
            return False, "Need target and direction"
        
        if action.target not in self.physics.objects:
            return False, "Object not found"
        
        obj = self.physics.objects[action.target]
        distance = self.position.distance_to(obj.position)
        
        if distance > self.reach_distance:
            return False, "Object too far to push"
        
        force = action.direction.normalize() * action.magnitude * 10.0
        self.physics.apply_force(action.target, force)
        
        return True, f"Pushed {obj.name}"
    
    def _execute_look(self, action: Action) -> Tuple[bool, str]:
        """Execute a look action."""
        if action.direction:
            self.orientation = math.atan2(action.direction.x, action.direction.z)
        return True, f"Looking at orientation {self.orientation:.2f}"


class EmbodimentInterface:
    """
    High-level interface for embodied cognition.
    
    Bridges the cognitive system with physical interaction.
    """
    
    def __init__(self):
        self.agent = EmbodiedAgent()
        self.action_history: List[ActionResult] = []
    
    def setup_world(self, objects: List[Dict[str, Any]]) -> None:
        """Set up the world with objects."""
        for obj_data in objects:
            obj = WorldObject(
                id=obj_data.get("id", str(len(self.agent.physics.objects))),
                name=obj_data.get("name", "unknown"),
                position=Vector3(
                    obj_data.get("x", 0),
                    obj_data.get("y", 0),
                    obj_data.get("z", 0),
                ),
                mass=obj_data.get("mass", 1.0),
                is_static=obj_data.get("static", False),
                is_graspable=obj_data.get("graspable", True),
            )
            self.agent.physics.add_object(obj)
    
    def observe(self) -> Dict[str, Any]:
        """Get current observation."""
        perception = self.agent.perceive()
        
        return {
            "visible_objects": [
                {
                    "id": o.id,
                    "name": o.name,
                    "distance": self.agent.position.distance_to(o.position),
                    "position": (o.position.x, o.position.y, o.position.z),
                }
                for o in perception.visible_objects
            ],
            "holding": self.agent.grasped_object,
            "energy": self.agent.energy,
            "position": (self.agent.position.x, self.agent.position.y, self.agent.position.z),
        }
    
    def act(self, action_type: str, **kwargs) -> Dict[str, Any]:
        """Execute an action by name."""
        action_map = {
            "move": ActionType.MOVE,
            "grasp": ActionType.GRASP,
            "release": ActionType.RELEASE,
            "push": ActionType.PUSH,
            "look": ActionType.LOOK,
        }
        
        if action_type not in action_map:
            return {"success": False, "error": f"Unknown action: {action_type}"}
        
        direction = None
        if "direction" in kwargs:
            d = kwargs["direction"]
            direction = Vector3(d[0], d[1], d[2]) if isinstance(d, (list, tuple)) else d
        
        action = Action(
            type=action_map[action_type],
            target=kwargs.get("target"),
            direction=direction,
            magnitude=kwargs.get("magnitude", 1.0),
        )
        
        result = self.agent.execute(action)
        self.action_history.append(result)
        
        return {
            "success": result.success,
            "feedback": result.feedback,
            "energy_cost": result.energy_cost,
            "observation": self.observe(),
        }
    
    def get_action_sequence(self, goal: str) -> List[Dict[str, Any]]:
        """Generate action sequence for a goal."""
        goal_lower = goal.lower()
        
        # Simple goal-to-action mapping
        if "pick up" in goal_lower or "grab" in goal_lower:
            # Find target object
            for word in goal_lower.split():
                obs = self.observe()
                for obj in obs["visible_objects"]:
                    if word in obj["name"].lower():
                        return [
                            {"action": "move", "direction": obj["position"]},
                            {"action": "grasp", "target": obj["id"]},
                        ]
        
        if "put down" in goal_lower or "release" in goal_lower:
            return [{"action": "release"}]
        
        if "push" in goal_lower:
            return [{"action": "push", "direction": [1, 0, 0], "magnitude": 2.0}]
        
        return []

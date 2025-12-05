"""Embodiment module - Physical grounding through simulation."""

from omniagi.embodiment.sim_env import SimulationEnvironment, PhysicsEngine
from omniagi.embodiment.sensory import SensoryProcessor, SensorReading
from omniagi.embodiment.actuator import ActuatorController, Action

__all__ = [
    "SimulationEnvironment",
    "PhysicsEngine",
    "SensoryProcessor",
    "SensorReading",
    "ActuatorController",
    "Action",
]

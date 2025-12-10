"""
OmniAGI Extensions System

Goose-style extensible architecture for adding capabilities.
"""

from omniagi.extensions.base import Extension, Tool, ExtensionManager
from omniagi.extensions.developer import DeveloperExtension
from omniagi.extensions.memory import MemoryExtension

__all__ = [
    "Extension",
    "Tool", 
    "ExtensionManager",
    "DeveloperExtension",
    "MemoryExtension",
]

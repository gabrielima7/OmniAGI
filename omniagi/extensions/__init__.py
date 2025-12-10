"""
OmniAGI Extensions System

Goose-style extensible architecture for adding capabilities.
"""

from omniagi.extensions.base import Extension, Tool, ExtensionManager, get_extension_manager
from omniagi.extensions.developer import DeveloperExtension
from omniagi.extensions.memory import MemoryExtension
from omniagi.extensions.web import WebExtension

__all__ = [
    "Extension",
    "Tool", 
    "ExtensionManager",
    "get_extension_manager",
    "DeveloperExtension",
    "MemoryExtension",
    "WebExtension",
]


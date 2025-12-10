"""
OmniAGI - Sistema Operacional Cognitivo
========================================

Inteligência Artificial Geral Soberana, Descentralizada e Autônoma.
"""

__version__ = "0.1.0"
__author__ = "OmniAGI Team"

# Lazy imports to avoid requiring pydantic for all modules
def get_config():
    from omniagi.core.config import get_config as _get_config
    return _get_config()

def Config():
    from omniagi.core.config import Config as _Config
    return _Config

def Engine():
    from omniagi.core.engine import Engine as _Engine
    return _Engine

__all__ = [
    "Config",
    "get_config", 
    "Engine",
    "__version__",
]


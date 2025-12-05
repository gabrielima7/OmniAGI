"""
OmniAGI - Sistema Operacional Cognitivo
========================================

Inteligência Artificial Geral Soberana, Descentralizada e Autônoma.
"""

__version__ = "0.1.0"
__author__ = "OmniAGI Team"

from omniagi.core.config import Config, get_config
from omniagi.core.engine import Engine

__all__ = [
    "Config",
    "get_config", 
    "Engine",
    "__version__",
]

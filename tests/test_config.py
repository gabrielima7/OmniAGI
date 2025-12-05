"""
Basic tests for OmniAGI core functionality.
"""

import pytest
from omniagi.core.config import Config, get_config


def test_config_creation():
    """Test that config can be created."""
    config = Config()
    assert config is not None
    assert config.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")


def test_config_singleton():
    """Test that get_config returns singleton."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_config_defaults():
    """Test default configuration values."""
    config = Config()
    assert config.model.context_length == 4096
    assert config.engine.device == "auto"
    assert config.security.sandbox_enabled is True

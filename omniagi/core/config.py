"""
Global configuration management for OmniAGI.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Model-specific configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OMNI_MODEL_")
    
    path: Path | None = Field(
        default=None,
        description="Path to the model file (GGUF or SafeTensors)"
    )
    context_length: int = Field(
        default=4096,
        description="Maximum context length"
    )
    gpu_layers: int = Field(
        default=0,
        description="Number of layers to offload to GPU (-1 = all)"
    )


class EngineConfig(BaseSettings):
    """Inference engine configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OMNI_ENGINE_")
    
    device: Literal["auto", "cpu", "cuda", "metal"] = Field(
        default="auto",
        description="Compute device to use"
    )
    threads: int = Field(
        default=0,
        description="Number of threads (0 = auto)"
    )
    batch_size: int = Field(
        default=512,
        description="Batch size for prompt processing"
    )


class MemoryConfig(BaseSettings):
    """Memory system configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OMNI_MEMORY_")
    
    working_memory_size: int = Field(
        default=10,
        description="Number of recent messages to keep in working memory"
    )
    vector_db_path: Path = Field(
        default=Path("~/.omniagi/memory/vectors").expanduser(),
        description="Path to vector database"
    )
    chunk_size: int = Field(
        default=512,
        description="Chunk size for text splitting"
    )


class DaemonConfig(BaseSettings):
    """Life daemon configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OMNI_DAEMON_")
    
    enabled: bool = Field(
        default=False,
        description="Enable autonomous life daemon"
    )
    introspection_interval: int = Field(
        default=300,
        description="Seconds between introspection cycles"
    )
    max_actions_per_cycle: int = Field(
        default=5,
        description="Maximum actions per life cycle"
    )


class SecurityConfig(BaseSettings):
    """Security and sandbox configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OMNI_SECURITY_")
    
    sandbox_enabled: bool = Field(
        default=True,
        description="Enable code execution sandbox"
    )
    execution_timeout: int = Field(
        default=30,
        description="Maximum seconds for code execution"
    )
    allowed_paths: list[str] = Field(
        default_factory=lambda: [str(Path.cwd())],
        description="Paths the agent can access"
    )
    blocked_imports: list[str] = Field(
        default_factory=lambda: ["os.system", "subprocess", "shutil.rmtree"],
        description="Blocked Python imports"
    )


class ServerConfig(BaseSettings):
    """API server configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OMNI_SERVER_")
    
    host: str = Field(
        default="127.0.0.1",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        description="Server port"
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication"
    )


class Config(BaseSettings):
    """Main OmniAGI configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="OMNI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    # General settings
    data_dir: Path = Field(
        default=Path("~/.omniagi").expanduser(),
        description="Base directory for OmniAGI data"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        dirs = [
            self.data_dir,
            self.data_dir / "models",
            self.data_dir / "memory",
            self.data_dir / "logs",
            self.data_dir / "sandbox",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_config() -> Config:
    """Get the global configuration singleton."""
    config = Config()
    config.ensure_dirs()
    return config

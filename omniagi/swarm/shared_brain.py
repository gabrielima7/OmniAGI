"""
Shared Brain - Centralized model access for the swarm.
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass, field
from typing import Any
from collections import deque

from omniagi.core.engine import Engine, GenerationConfig, GenerationOutput

logger = structlog.get_logger()


@dataclass
class InferenceRequest:
    """A request for inference."""
    
    id: str
    prompt: str | list[dict[str, str]]
    config: GenerationConfig
    is_chat: bool = False
    future: asyncio.Future = field(default_factory=asyncio.Future)


class SharedBrain:
    """
    Shared inference engine for the swarm.
    
    All agents in the swarm share access to a single model instance,
    preventing memory duplication and enabling efficient resource usage.
    
    Features:
    - Request queuing
    - Concurrent access management
    - Batching (future optimization)
    """
    
    def __init__(self, engine: Engine):
        """
        Initialize shared brain.
        
        Args:
            engine: The inference engine instance.
        """
        self.engine = engine
        self._queue: deque[InferenceRequest] = deque()
        self._lock = asyncio.Lock()
        self._processing = False
        self._request_counter = 0
        
        logger.info("Shared brain initialized")
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.engine.is_loaded
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        """
        Generate text using the shared model.
        
        Args:
            prompt: The input prompt.
            config: Generation configuration.
            
        Returns:
            GenerationOutput from the model.
        """
        config = config or GenerationConfig()
        
        request = InferenceRequest(
            id=f"req_{self._request_counter}",
            prompt=prompt,
            config=config,
            is_chat=False,
            future=asyncio.get_event_loop().create_future(),
        )
        self._request_counter += 1
        
        self._queue.append(request)
        await self._process_queue()
        
        return await request.future
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        """
        Chat completion using the shared model.
        
        Args:
            messages: Chat messages.
            config: Generation configuration.
            
        Returns:
            GenerationOutput from the model.
        """
        config = config or GenerationConfig()
        
        request = InferenceRequest(
            id=f"req_{self._request_counter}",
            prompt=messages,
            config=config,
            is_chat=True,
            future=asyncio.get_event_loop().create_future(),
        )
        self._request_counter += 1
        
        self._queue.append(request)
        await self._process_queue()
        
        return await request.future
    
    async def _process_queue(self) -> None:
        """Process pending requests in the queue."""
        async with self._lock:
            if self._processing:
                return
            self._processing = True
        
        try:
            while self._queue:
                request = self._queue.popleft()
                
                try:
                    if request.is_chat:
                        result = self.engine.chat(request.prompt, request.config)
                    else:
                        result = self.engine.generate(request.prompt, request.config)
                    
                    request.future.set_result(result)
                    
                except Exception as e:
                    logger.error("Inference failed", request_id=request.id, error=str(e))
                    request.future.set_exception(e)
        
        finally:
            async with self._lock:
                self._processing = False
    
    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._request_counter,
            "queue_size": len(self._queue),
            "model_loaded": self.is_loaded,
        }

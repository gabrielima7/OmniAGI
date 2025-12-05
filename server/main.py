"""
OmniAGI API Server - OpenAI-compatible API.
"""

import time
import uuid
import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from omniagi.core.config import get_config
from omniagi.core.engine import Engine, GenerationConfig

logger = structlog.get_logger()


# Request/Response Models
class Message(BaseModel):
    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512
    stop: list[str] | None = None
    stream: bool = False


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop: list[str] | None = None
    stream: bool = False


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "omniagi"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# Global engine instance
_engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    global _engine
    
    config = get_config()
    _engine = Engine()
    
    if config.model.path:
        logger.info("Loading model", path=str(config.model.path))
        try:
            _engine.load()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
    else:
        logger.warning("No model path configured")
    
    yield
    
    if _engine:
        _engine.unload()
        logger.info("Engine unloaded")


# Create app
app = FastAPI(
    title="OmniAGI API",
    description="OpenAI-compatible API for OmniAGI",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> Engine:
    """Dependency to get the engine."""
    if _engine is None or not _engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Configure model.path and restart server."
        )
    return _engine


async def verify_api_key(
    authorization: str | None = Header(None),
) -> None:
    """Verify API key if configured."""
    config = get_config()
    
    if config.server.api_key:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing API key")
        
        # Extract Bearer token
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization format")
        
        if parts[1] != config.server.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")


# Routes
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "OmniAGI API",
        "version": "0.1.0",
        "status": "ok",
    }


@app.get("/v1/models", response_model=ModelList)
async def list_models(
    engine: Engine = Depends(get_engine),
    _: None = Depends(verify_api_key),
):
    """List available models."""
    model_name = engine.model_path.stem if engine.model_path else "unknown"
    
    return ModelList(
        data=[
            ModelInfo(id=model_name),
            ModelInfo(id="local"),  # Alias
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    engine: Engine = Depends(get_engine),
    _: None = Depends(verify_api_key),
):
    """Create a chat completion."""
    logger.info(
        "Chat completion request",
        model=request.model,
        messages=len(request.messages),
    )
    
    # Convert to engine format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop or [],
    )
    
    try:
        output = engine.chat(messages, config)
    except Exception as e:
        logger.error("Chat completion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=output.text),
                finish_reason="stop" if not output.stopped_by_length else "length",
            )
        ],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.tokens_generated,
            total_tokens=output.prompt_tokens + output.tokens_generated,
        ),
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    engine: Engine = Depends(get_engine),
    _: None = Depends(verify_api_key),
):
    """Create a text completion."""
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    
    logger.info(
        "Completion request",
        model=request.model,
        prompt_length=len(prompt),
    )
    
    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop or [],
    )
    
    try:
        output = engine.generate(prompt, config)
    except Exception as e:
        logger.error("Completion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
    return CompletionResponse(
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                text=output.text,
                finish_reason="stop" if not output.stopped_by_length else "length",
            )
        ],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.tokens_generated,
            total_tokens=output.prompt_tokens + output.tokens_generated,
        ),
    )


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": _engine is not None and _engine.is_loaded,
        "model_path": str(_engine.model_path) if _engine and _engine.model_path else None,
    }

"""
External API Integration.

Connects OmniAGI to external services like:
- OpenAI/Claude/Gemini for LLM capabilities
- Web search APIs
- Knowledge bases
"""

from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


@dataclass
class APIResponse:
    """Response from an API call."""
    success: bool
    data: Any
    error: Optional[str] = None
    latency_ms: int = 0


class BaseAPIClient(ABC):
    """Base class for API clients."""
    
    @abstractmethod
    def call(self, endpoint: str, **kwargs) -> APIResponse:
        """Make an API call."""
        pass


class OpenAIClient(BaseAPIClient):
    """
    OpenAI API client.
    
    Supports chat completions and embeddings.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.default_model = "gpt-4o-mini"
    
    def call(self, endpoint: str, **kwargs) -> APIResponse:
        """Make an API call."""
        if not HTTPX_AVAILABLE:
            return APIResponse(False, None, "httpx not available")
        
        if not self.api_key:
            return APIResponse(False, None, "API key not set")
        
        import time
        start = time.time()
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/{endpoint}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=kwargs,
                    timeout=60.0,
                )
                
                latency = int((time.time() - start) * 1000)
                
                if response.status_code == 200:
                    return APIResponse(True, response.json(), latency_ms=latency)
                else:
                    return APIResponse(False, None, response.text, latency)
                    
        except Exception as e:
            return APIResponse(False, None, str(e))
    
    def chat(self, messages: List[Dict], model: str = None) -> APIResponse:
        """Send chat completion request."""
        return self.call(
            "chat/completions",
            model=model or self.default_model,
            messages=messages,
        )
    
    def complete(self, prompt: str, model: str = None) -> str:
        """Simple completion."""
        result = self.chat([{"role": "user", "content": prompt}], model)
        if result.success:
            return result.data["choices"][0]["message"]["content"]
        return f"Error: {result.error}"
    
    def embed(self, text: str, model: str = "text-embedding-3-small") -> APIResponse:
        """Get text embeddings."""
        return self.call(
            "embeddings",
            model=model,
            input=text,
        )


class AnthropicClient(BaseAPIClient):
    """
    Anthropic (Claude) API client.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1"
        self.default_model = "claude-3-haiku-20240307"
    
    def call(self, endpoint: str, **kwargs) -> APIResponse:
        """Make an API call."""
        if not HTTPX_AVAILABLE:
            return APIResponse(False, None, "httpx not available")
        
        if not self.api_key:
            return APIResponse(False, None, "API key not set")
        
        import time
        start = time.time()
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/{endpoint}",
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=kwargs,
                    timeout=60.0,
                )
                
                latency = int((time.time() - start) * 1000)
                
                if response.status_code == 200:
                    return APIResponse(True, response.json(), latency_ms=latency)
                else:
                    return APIResponse(False, None, response.text, latency)
                    
        except Exception as e:
            return APIResponse(False, None, str(e))
    
    def chat(self, messages: List[Dict], model: str = None) -> APIResponse:
        """Send chat request."""
        return self.call(
            "messages",
            model=model or self.default_model,
            max_tokens=1024,
            messages=messages,
        )
    
    def complete(self, prompt: str, model: str = None) -> str:
        """Simple completion."""
        result = self.chat([{"role": "user", "content": prompt}], model)
        if result.success:
            return result.data["content"][0]["text"]
        return f"Error: {result.error}"


class GeminiClient(BaseAPIClient):
    """
    Google Gemini API client.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.default_model = "gemini-1.5-flash"
    
    def call(self, endpoint: str, **kwargs) -> APIResponse:
        """Make an API call."""
        if not HTTPX_AVAILABLE:
            return APIResponse(False, None, "httpx not available")
        
        if not self.api_key:
            return APIResponse(False, None, "API key not set")
        
        import time
        start = time.time()
        
        try:
            with httpx.Client() as client:
                url = f"{self.base_url}/{endpoint}?key={self.api_key}"
                response = client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=kwargs,
                    timeout=60.0,
                )
                
                latency = int((time.time() - start) * 1000)
                
                if response.status_code == 200:
                    return APIResponse(True, response.json(), latency_ms=latency)
                else:
                    return APIResponse(False, None, response.text, latency)
                    
        except Exception as e:
            return APIResponse(False, None, str(e))
    
    def generate(self, prompt: str, model: str = None) -> APIResponse:
        """Generate content."""
        model = model or self.default_model
        return self.call(
            f"models/{model}:generateContent",
            contents=[{"parts": [{"text": prompt}]}],
        )
    
    def complete(self, prompt: str, model: str = None) -> str:
        """Simple completion."""
        result = self.generate(prompt, model)
        if result.success:
            candidates = result.data.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"]
        return f"Error: {result.error}"


class WebSearchClient:
    """
    Web search using DuckDuckGo (no API key needed).
    """
    
    def __init__(self):
        self.search_url = "https://html.duckduckgo.com/html/"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the web."""
        if not HTTPX_AVAILABLE:
            return []
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    self.search_url,
                    data={"q": query},
                    headers={"User-Agent": "OmniAGI/1.0"},
                    timeout=10.0,
                )
                
                # Parse HTML results (simplified)
                results = []
                # In production, use BeautifulSoup
                if "result" in response.text.lower():
                    results.append({
                        "title": f"Search results for: {query}",
                        "snippet": response.text[:500],
                        "url": f"https://duckduckgo.com/?q={query}",
                    })
                
                return results[:max_results]
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


class APIManager:
    """
    Manages all external API connections.
    
    Provides a unified interface for AGI to access external capabilities.
    """
    
    def __init__(self):
        self.openai = OpenAIClient()
        self.anthropic = AnthropicClient()
        self.gemini = GeminiClient()
        self.search = WebSearchClient()
    
    def complete(self, prompt: str, provider: str = "auto") -> str:
        """
        Complete a prompt using best available provider.
        
        provider: "openai", "anthropic", "gemini", or "auto"
        """
        if provider == "auto":
            # Try providers in order of preference
            for client in [self.openai, self.anthropic, self.gemini]:
                if client.api_key:
                    return client.complete(prompt)
            return "No API keys configured"
        
        elif provider == "openai":
            return self.openai.complete(prompt)
        elif provider == "anthropic":
            return self.anthropic.complete(prompt)
        elif provider == "gemini":
            return self.gemini.complete(prompt)
        
        return "Unknown provider"
    
    def search_web(self, query: str) -> List[Dict]:
        """Search the web."""
        return self.search.search(query)
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get available providers."""
        return {
            "openai": bool(self.openai.api_key),
            "anthropic": bool(self.anthropic.api_key),
            "gemini": bool(self.gemini.api_key),
            "search": HTTPX_AVAILABLE,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API stats."""
        return {
            "httpx_available": HTTPX_AVAILABLE,
            "providers": self.get_available_providers(),
        }

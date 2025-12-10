"""
Web Extension

Provides HTTP and web scraping tools.
Inspired by Goose's Computer Controller extension.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urljoin, urlparse

from omniagi.extensions.base import Extension, Tool

# Make structlog optional
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)


# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False


class WebExtension(Extension):
    """
    Web extension for HTTP requests and web scraping.
    
    Tools:
    - http_get: Make HTTP GET request
    - http_post: Make HTTP POST request
    - scrape_text: Extract text from webpage
    - scrape_links: Extract links from webpage
    """
    
    name = "web"
    description = "HTTP requests and web scraping tools"
    version = "1.0.0"
    
    def __init__(self, timeout: int = 30):
        super().__init__()
        self.timeout = timeout
        self._session = None
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register web tools."""
        self.register_tool(Tool(
            name="http_get",
            description="Make an HTTP GET request to a URL",
            handler=self._http_get,
            parameters={
                "url": {"type": "string", "description": "URL to request"},
                "headers": {"type": "object", "description": "Request headers", "default": {}},
            },
        ))
        
        self.register_tool(Tool(
            name="http_post",
            description="Make an HTTP POST request to a URL",
            handler=self._http_post,
            parameters={
                "url": {"type": "string", "description": "URL to request"},
                "data": {"type": "object", "description": "Request body", "default": {}},
                "headers": {"type": "object", "description": "Request headers", "default": {}},
            },
        ))
        
        self.register_tool(Tool(
            name="scrape_text",
            description="Extract text content from a webpage",
            handler=self._scrape_text,
            parameters={
                "url": {"type": "string", "description": "URL to scrape"},
                "selector": {"type": "string", "description": "CSS selector", "default": "body"},
            },
        ))
        
        self.register_tool(Tool(
            name="scrape_links",
            description="Extract all links from a webpage",
            handler=self._scrape_links,
            parameters={
                "url": {"type": "string", "description": "URL to scrape"},
            },
        ))
        
        self.register_tool(Tool(
            name="download",
            description="Download a file from URL",
            handler=self._download,
            parameters={
                "url": {"type": "string", "description": "URL to download"},
                "path": {"type": "string", "description": "Local path to save file"},
            },
            requires_confirmation=True,
        ))
    
    def _setup(self) -> None:
        """Initialize session."""
        if REQUESTS_AVAILABLE:
            self._session = requests.Session()
    
    def _cleanup(self) -> None:
        """Close session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def _check_requests(self) -> dict | None:
        """Check if requests is available."""
        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "error": "requests library not installed. Run: pip install requests",
            }
        return None
    
    def _check_bs4(self) -> dict | None:
        """Check if BeautifulSoup is available."""
        if not BS4_AVAILABLE:
            return {
                "success": False,
                "error": "beautifulsoup4 not installed. Run: pip install beautifulsoup4",
            }
        return None
    
    def _http_get(self, url: str, headers: dict = None) -> dict:
        """Make HTTP GET request."""
        if err := self._check_requests():
            return err
        
        headers = headers or {}
        logger.info("HTTP GET", url=url)
        
        try:
            session = self._session or requests
            response = session.get(url, headers=headers, timeout=self.timeout)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:5000],  # Limit content size
                "content_length": len(response.text),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _http_post(self, url: str, data: dict = None, headers: dict = None) -> dict:
        """Make HTTP POST request."""
        if err := self._check_requests():
            return err
        
        data = data or {}
        headers = headers or {}
        logger.info("HTTP POST", url=url)
        
        try:
            session = self._session or requests
            response = session.post(
                url, 
                json=data, 
                headers=headers, 
                timeout=self.timeout
            )
            
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:5000],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _scrape_text(self, url: str, selector: str = "body") -> dict:
        """Extract text from webpage."""
        if err := self._check_requests():
            return err
        if err := self._check_bs4():
            return err
        
        logger.info("Scraping text", url=url, selector=selector)
        
        try:
            session = self._session or requests
            response = session.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.text, "html.parser")
            
            elements = soup.select(selector)
            texts = [el.get_text(strip=True) for el in elements]
            
            return {
                "success": True,
                "url": url,
                "selector": selector,
                "texts": texts,
                "count": len(texts),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _scrape_links(self, url: str) -> dict:
        """Extract all links from webpage."""
        if err := self._check_requests():
            return err
        if err := self._check_bs4():
            return err
        
        logger.info("Scraping links", url=url)
        
        try:
            session = self._session or requests
            response = session.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.text, "html.parser")
            
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # Convert relative URLs to absolute
                absolute_url = urljoin(url, href)
                links.append({
                    "text": a.get_text(strip=True),
                    "href": absolute_url,
                })
            
            return {
                "success": True,
                "url": url,
                "links": links[:100],  # Limit to 100 links
                "count": len(links),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _download(self, url: str, path: str) -> dict:
        """Download file from URL."""
        if err := self._check_requests():
            return err
        
        from pathlib import Path
        logger.info("Downloading", url=url, path=path)
        
        try:
            session = self._session or requests
            response = session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "success": True,
                "url": url,
                "path": str(filepath),
                "size": filepath.stat().st_size,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

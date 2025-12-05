"""
Web tools for searching and crawling.
"""

from __future__ import annotations

import re
import structlog
import aiohttp
from typing import Any
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from omniagi.tools.base import Tool, ToolResult

logger = structlog.get_logger()


def html_to_markdown(html: str, base_url: str = "") -> str:
    """Convert HTML to simplified markdown."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    
    # Get text
    text = soup.get_text(separator="\n", strip=True)
    
    # Clean up whitespace
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    
    return "\n\n".join(lines)


class WebSearchTool(Tool):
    """
    Tool for web search using DuckDuckGo (no API key required).
    """
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo. Returns top results with titles and snippets."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "query": {
                "description": "Search query",
                "type": "string",
                "required": True,
            },
            "max_results": {
                "description": "Maximum number of results (default: 5)",
                "type": "integer",
                "required": False,
            },
        }
    
    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            # Use DuckDuckGo HTML search (no API needed)
            url = "https://html.duckduckgo.com/html/"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data={"q": query},
                    headers={"User-Agent": "OmniAGI/1.0"},
                ) as response:
                    if response.status != 200:
                        return ToolResult.error(f"Search failed: HTTP {response.status}")
                    
                    html = await response.text()
            
            soup = BeautifulSoup(html, "html.parser")
            results = []
            
            for result in soup.select(".result")[:max_results]:
                title_elem = result.select_one(".result__title")
                snippet_elem = result.select_one(".result__snippet")
                link_elem = result.select_one(".result__url")
                
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    link = link_elem.get_text(strip=True) if link_elem else ""
                    
                    results.append(f"**{title}**\n{link}\n{snippet}")
            
            if not results:
                return ToolResult.success("No results found.")
            
            output = "\n\n---\n\n".join(results)
            logger.info("Web search completed", query=query, results=len(results))
            return ToolResult.success(output, query=query, count=len(results))
            
        except Exception as e:
            logger.error("Web search failed", query=query, error=str(e))
            return ToolResult.error(str(e))


class WebCrawlerTool(Tool):
    """
    Tool for crawling and extracting content from web pages.
    """
    
    @property
    def name(self) -> str:
        return "web_crawler"
    
    @property
    def description(self) -> str:
        return "Fetch and extract text content from a web page URL."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "url": {
                "description": "URL of the page to crawl",
                "type": "string",
                "required": True,
            },
            "max_length": {
                "description": "Maximum content length (default: 10000)",
                "type": "integer",
                "required": False,
            },
        }
    
    async def execute(self, url: str, max_length: int = 10000) -> ToolResult:
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme in ("http", "https"):
                return ToolResult.error(f"Invalid URL scheme: {parsed.scheme}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"User-Agent": "OmniAGI/1.0"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        return ToolResult.error(f"Failed to fetch: HTTP {response.status}")
                    
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" not in content_type and "text/plain" not in content_type:
                        return ToolResult.error(f"Unsupported content type: {content_type}")
                    
                    html = await response.text()
            
            # Convert to markdown-like text
            text = html_to_markdown(html, url)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[Content truncated...]"
            
            logger.info("Web page crawled", url=url, length=len(text))
            return ToolResult.success(text, url=url, length=len(text))
            
        except aiohttp.ClientError as e:
            logger.error("Failed to crawl URL", url=url, error=str(e))
            return ToolResult.error(f"Network error: {str(e)}")
        except Exception as e:
            logger.error("Failed to crawl URL", url=url, error=str(e))
            return ToolResult.error(str(e))

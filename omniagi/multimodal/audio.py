"""
Audio processing - Speech-to-text and audio analysis.
"""

from __future__ import annotations

import structlog
from pathlib import Path
from typing import Any

logger = structlog.get_logger()


class AudioProcessor:
    """
    Process and transcribe audio.
    
    Uses Whisper for speech-to-text transcription.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize audio processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
        """
        self.model_size = model_size
        self._model = None
    
    def _load_model(self):
        """Load the Whisper model (lazy loading)."""
        if self._model is None:
            try:
                import whisper
                logger.info("Loading Whisper model", size=self.model_size)
                self._model = whisper.load_model(self.model_size)
                logger.info("Whisper model loaded")
            except ImportError:
                logger.warning("Whisper not installed. Install with: pip install openai-whisper")
                raise ImportError(
                    "openai-whisper not installed. "
                    "Install with: pip install 'omniagi[audio]'"
                )
    
    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file.
            language: Language code (e.g., 'en', 'pt'). None for auto-detect.
            
        Returns:
            Dictionary with transcription results.
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info("Transcribing audio", path=str(audio_path))
        
        options = {}
        if language:
            options["language"] = language
        
        result = self._model.transcribe(str(audio_path), **options)
        
        logger.info(
            "Transcription complete",
            language=result.get("language"),
            duration=len(result.get("segments", [])),
        )
        
        return {
            "text": result["text"],
            "language": result.get("language"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in result.get("segments", [])
            ],
        }
    
    def transcribe_to_text(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> str:
        """
        Simple transcription returning just the text.
        
        Args:
            audio_path: Path to audio file.
            language: Language code.
            
        Returns:
            Transcribed text.
        """
        result = self.transcribe(audio_path, language)
        return result["text"]
    
    async def transcribe_async(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Async wrapper for transcription."""
        import asyncio
        return await asyncio.to_thread(self.transcribe, audio_path, language)

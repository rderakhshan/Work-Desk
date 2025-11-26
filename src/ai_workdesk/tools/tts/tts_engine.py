"""
Text-to-Speech engine module.

This module provides CPU-friendly text-to-speech conversion using
pyttsx3 (system TTS) as the primary engine and gTTS (Google TTS) as
a fallback option for better quality.
"""

from typing import List, Optional, Dict
from pathlib import Path
import os
from loguru import logger

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed. System TTS will not be available.")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS not installed. Google TTS will not be available.")


class TTSEngine:
    """
    Text-to-Speech engine with multiple backend support.
    
    Supports:
    - pyttsx3: System TTS (instant, offline, CPU-friendly)
    - gTTS: Google TTS (cloud-based, better quality, requires internet)
    """
    
    def __init__(
        self,
        engine: str = "pyttsx3",
        voice_id: Optional[str] = None,
        rate: int = 150,
        volume: float = 1.0
    ):
        """
        Initialize the TTS engine.
        
        Args:
            engine: TTS engine to use ('pyttsx3' or 'gtts')
            voice_id: Optional voice ID for pyttsx3
            rate: Speech rate in words per minute (pyttsx3 only)
            volume: Volume level 0.0 to 1.0 (pyttsx3 only)
        """
        self.engine_type = engine.lower()
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.engine = None
        
        if self.engine_type == "pyttsx3":
            if not PYTTSX3_AVAILABLE:
                raise ImportError(
                    "pyttsx3 is not installed. "
                    "Install it with: uv sync --extra audio"
                )
            self._init_pyttsx3()
        elif self.engine_type == "gtts":
            if not GTTS_AVAILABLE:
                raise ImportError(
                    "gTTS is not installed. "
                    "Install it with: uv sync --extra audio"
                )
            logger.info("Using gTTS (Google Text-to-Speech)")
        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")
        
        logger.info(f"TTSEngine initialized with engine={self.engine_type}")
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine."""
        try:
            self.engine = pyttsx3.init()
            
            # Set rate
            self.engine.setProperty('rate', self.rate)
            
            # Set volume
            self.engine.setProperty('volume', self.volume)
            
            # Set voice if specified
            if self.voice_id:
                self.engine.setProperty('voice', self.voice_id)
            
            logger.info("pyttsx3 engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise
    
    def list_voices(self) -> List[Dict[str, str]]:
        """
        List available voices (pyttsx3 only).
        
        Returns:
            List of dictionaries with voice information
        """
        if self.engine_type != "pyttsx3":
            logger.warning("Voice listing only available for pyttsx3")
            return []
        
        if self.engine is None:
            self._init_pyttsx3()
        
        voices = self.engine.getProperty('voices')
        voice_list = []
        
        for voice in voices:
            voice_info = {
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages if hasattr(voice, 'languages') else [],
                'gender': voice.gender if hasattr(voice, 'gender') else 'unknown'
            }
            voice_list.append(voice_info)
        
        logger.info(f"Found {len(voice_list)} available voices")
        return voice_list
    
    def set_voice(self, voice_id: str):
        """
        Set the voice to use (pyttsx3 only).
        
        Args:
            voice_id: Voice ID from list_voices()
        """
        if self.engine_type != "pyttsx3":
            logger.warning("Voice selection only available for pyttsx3")
            return
        
        if self.engine is None:
            self._init_pyttsx3()
        
        try:
            self.engine.setProperty('voice', voice_id)
            self.voice_id = voice_id
            logger.info(f"Voice set to: {voice_id}")
        except Exception as e:
            logger.error(f"Failed to set voice: {e}")
    
    def set_rate(self, rate: int):
        """
        Set speech rate (pyttsx3 only).
        
        Args:
            rate: Speech rate in words per minute (typically 100-200)
        """
        if self.engine_type != "pyttsx3":
            logger.warning("Rate control only available for pyttsx3")
            return
        
        if self.engine is None:
            self._init_pyttsx3()
        
        self.engine.setProperty('rate', rate)
        self.rate = rate
        logger.info(f"Speech rate set to: {rate} wpm")
    
    def set_volume(self, volume: float):
        """
        Set volume level (pyttsx3 only).
        
        Args:
            volume: Volume level from 0.0 to 1.0
        """
        if self.engine_type != "pyttsx3":
            logger.warning("Volume control only available for pyttsx3")
            return
        
        if self.engine is None:
            self._init_pyttsx3()
        
        volume = max(0.0, min(1.0, volume))  # Clamp to 0-1
        self.engine.setProperty('volume', volume)
        self.volume = volume
        logger.info(f"Volume set to: {volume}")
    
    def speak(
        self,
        text: str,
        save_path: Optional[str] = None,
        lang: str = 'en',
        slow: bool = False
    ) -> Optional[str]:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            save_path: Optional path to save audio file
            lang: Language code for gTTS (e.g., 'en', 'es', 'fr')
            slow: Slow speech for gTTS
            
        Returns:
            Path to saved file if save_path provided, None otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        logger.info(f"Converting text to speech: {len(text)} characters")
        
        try:
            if self.engine_type == "pyttsx3":
                return self._speak_pyttsx3(text, save_path)
            elif self.engine_type == "gtts":
                return self._speak_gtts(text, save_path, lang, slow)
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
    
    def _speak_pyttsx3(self, text: str, save_path: Optional[str] = None) -> Optional[str]:
        """Generate speech using pyttsx3."""
        if self.engine is None:
            self._init_pyttsx3()
        
        if save_path:
            # Save to file
            self.engine.save_to_file(text, save_path)
            self.engine.runAndWait()
            logger.info(f"Audio saved to: {save_path}")
            return save_path
        else:
            # Speak directly
            self.engine.say(text)
            self.engine.runAndWait()
            logger.info("Speech completed")
            return None
    
    def _speak_gtts(
        self,
        text: str,
        save_path: Optional[str] = None,
        lang: str = 'en',
        slow: bool = False
    ) -> Optional[str]:
        """Generate speech using gTTS."""
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        if save_path:
            # Save to specified path
            tts.save(save_path)
            logger.info(f"Audio saved to: {save_path}")
            return save_path
        else:
            # Save to temporary file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()
            
            tts.save(temp_path)
            logger.info(f"Audio saved to temporary file: {temp_path}")
            return temp_path
    
    def speak_batch(
        self,
        texts: List[str],
        output_dir: str,
        prefix: str = "speech",
        lang: str = 'en'
    ) -> List[str]:
        """
        Convert multiple texts to speech files.
        
        Args:
            texts: List of texts to convert
            output_dir: Directory to save audio files
            prefix: Prefix for output filenames
            lang: Language code for gTTS
            
        Returns:
            List of paths to generated audio files
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Batch TTS: {len(texts)} texts")
        
        output_paths = []
        for i, text in enumerate(texts, 1):
            output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.mp3")
            try:
                self.speak(text, save_path=output_path, lang=lang)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to generate speech for text {i}: {e}")
                output_paths.append(None)
        
        logger.info(f"Batch TTS complete: {len(output_paths)} files generated")
        return output_paths
    
    def get_engine_info(self) -> Dict[str, any]:
        """
        Get information about the TTS engine.
        
        Returns:
            Dictionary with engine information
        """
        info = {
            'engine_type': self.engine_type,
            'rate': self.rate if self.engine_type == 'pyttsx3' else None,
            'volume': self.volume if self.engine_type == 'pyttsx3' else None,
            'voice_id': self.voice_id if self.engine_type == 'pyttsx3' else None,
        }
        
        if self.engine_type == 'pyttsx3':
            voices = self.list_voices()
            info['available_voices'] = len(voices)
        
        return info

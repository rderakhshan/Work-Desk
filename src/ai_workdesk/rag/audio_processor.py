"""
Audio processing module for speech-to-text transcription.

This module provides CPU-optimized audio transcription using faster-whisper,
designed for efficient processing on systems without GPU acceleration.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
from loguru import logger

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Audio processing will not be available.")

try:
    import soundfile as sf
except ImportError:
    sf = None
    logger.warning("soundfile not installed. Audio duration detection may be limited.")


class AudioProcessor:
    """
    Handles audio file processing and transcription using faster-whisper.
    
    Optimized for CPU-only environments with efficient memory usage and
    processing speed. Supports multiple audio formats and provides both
    simple transcription and timestamp-aware transcription.
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.mp4'}
    
    def __init__(
        self, 
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize the audio processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu or cuda)
            compute_type: Computation type (int8, float16, float32)
                         int8 is recommended for CPU
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: uv sync --extra audio"
            )
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        logger.info(f"AudioProcessor initialized with model={model_size}, device={device}")
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate if the file is a supported audio format.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported audio format: {file_ext}")
            logger.info(f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
            return False
        
        return True
    
    def get_audio_duration(self, file_path: str) -> Optional[float]:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Duration in seconds, or None if unable to determine
        """
        if sf is None:
            logger.warning("soundfile not available, cannot determine audio duration")
            return None
        
        try:
            info = sf.info(file_path)
            duration = info.duration
            logger.info(f"Audio duration: {duration:.2f} seconds")
            return duration
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return None
    
    def transcribe_audio(
        self, 
        file_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            file_path: Path to the audio file
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            task: 'transcribe' or 'translate' (translate to English)
            
        Returns:
            Transcribed text as a single string
        """
        # Validate file
        if not self.validate_audio_file(file_path):
            raise ValueError(f"Invalid audio file: {file_path}")
        
        # Load model if not already loaded
        self._load_model()
        
        # Get duration for logging
        duration = self.get_audio_duration(file_path)
        if duration:
            logger.info(f"Transcribing {duration:.2f}s audio file: {file_path}")
        else:
            logger.info(f"Transcribing audio file: {file_path}")
        
        try:
            # Transcribe
            segments, info = self.model.transcribe(
                file_path,
                language=language,
                task=task,
                beam_size=5,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Detected language
            detected_lang = info.language
            logger.info(f"Detected language: {detected_lang} (probability: {info.language_probability:.2f})")
            
            # Combine all segments
            full_text = " ".join([segment.text.strip() for segment in segments])
            
            logger.info(f"Transcription complete: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_with_timestamps(
        self,
        file_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Tuple[str, List[Dict]]:
        """
        Transcribe an audio file with timestamp information.
        
        Args:
            file_path: Path to the audio file
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            task: 'transcribe' or 'translate' (translate to English)
            
        Returns:
            Tuple of (full_text, segments_list)
            segments_list contains dicts with 'text', 'start', 'end' keys
        """
        # Validate file
        if not self.validate_audio_file(file_path):
            raise ValueError(f"Invalid audio file: {file_path}")
        
        # Load model if not already loaded
        self._load_model()
        
        logger.info(f"Transcribing with timestamps: {file_path}")
        
        try:
            # Transcribe
            segments, info = self.model.transcribe(
                file_path,
                language=language,
                task=task,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Detected language
            detected_lang = info.language
            logger.info(f"Detected language: {detected_lang}")
            
            # Process segments
            segments_list = []
            full_text_parts = []
            
            for segment in segments:
                segment_dict = {
                    'text': segment.text.strip(),
                    'start': segment.start,
                    'end': segment.end,
                    'language': detected_lang
                }
                segments_list.append(segment_dict)
                full_text_parts.append(segment.text.strip())
            
            full_text = " ".join(full_text_parts)
            
            logger.info(f"Transcription complete: {len(segments_list)} segments, {len(full_text)} characters")
            return full_text, segments_list
            
        except Exception as e:
            logger.error(f"Transcription with timestamps failed: {e}")
            raise
    
    def transcribe_batch(
        self,
        file_paths: List[str],
        language: Optional[str] = None
    ) -> List[str]:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            file_paths: List of paths to audio files
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            
        Returns:
            List of transcribed texts
        """
        logger.info(f"Batch transcribing {len(file_paths)} files")
        
        results = []
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing file {i}/{len(file_paths)}: {file_path}")
            try:
                text = self.transcribe_audio(file_path, language=language)
                results.append(text)
            except Exception as e:
                logger.error(f"Failed to transcribe {file_path}: {e}")
                results.append(f"[Transcription failed: {str(e)}]")
        
        logger.info(f"Batch transcription complete: {len(results)} files processed")
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'model_loaded': self.model is not None,
            'supported_formats': ', '.join(self.SUPPORTED_FORMATS)
        }

"""
YouTube video transcript loader with advanced features.

This module provides comprehensive YouTube video processing including:
- Transcript extraction with language fallback
- Rich metadata extraction (title, channel, duration, etc.)
- Timestamp-aware chunking for precise citations
- Chapter detection and preservation
- Playlist expansion support
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from loguru import logger
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
import yt_dlp
from langchain_core.documents import Document


class YouTubeTranscriptLoader:
    """
    Loads and processes YouTube video transcripts for RAG ingestion.
    
    Features:
    - Multi-format URL parsing
    - Transcript fetching with automatic fallback
    - Rich metadata extraction
    - Timestamp preservation for citations
    - Chapter awareness
    - Playlist support
    """
    
    def __init__(self):
        """Initialize the YouTube transcript loader."""
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
    
    def _parse_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats.
        
        Supported formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://m.youtube.com/watch?v=VIDEO_ID
        - https://youtube.com/embed/VIDEO_ID
        - https://youtube.com/v/VIDEO_ID
        
        Args:
            url: YouTube URL string
            
        Returns:
            Video ID string or None if invalid
        """
        # Handle youtu.be short links
        if 'youtu.be/' in url:
            match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', url)
            return match.group(1) if match else None
        
        # Handle standard youtube.com URLs
        if 'youtube.com' in url or 'youtube-nocookie.com' in url:
            # Try to parse query parameters
            parsed = urlparse(url)
            
            # Check for v= parameter
            if 'v=' in url:
                params = parse_qs(parsed.query)
                if 'v' in params:
                    return params['v'][0]
            
            # Check for embed or v path
            match = re.search(r'(?:embed|v)/([a-zA-Z0-9_-]{11})', url)
            if match:
                return match.group(1)
        
        # Try to extract 11-character video ID directly
        match = re.search(r'([a-zA-Z0-9_-]{11})', url)
        return match.group(1) if match else None
    
    def _is_playlist_url(self, url: str) -> bool:
        """
        Check if URL is a playlist URL.
        
        Args:
            url: YouTube URL string
            
        Returns:
            True if playlist URL, False otherwise
        """
        return 'list=' in url and 'watch?' not in url
    
    def extract_playlist_videos(self, playlist_url: str) -> List[str]:
        """
        Extract all video URLs from a YouTube playlist.
        
        Args:
            playlist_url: YouTube playlist URL
            
        Returns:
            List of video URLs
        """
        try:
            logger.info(f"Extracting videos from playlist: {playlist_url}")
            
            opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,  # Don't download, just get URLs
                'playlistend': 100,  # Limit to first 100 videos
            }
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                
                if 'entries' in info:
                    video_urls = [
                        f"https://www.youtube.com/watch?v={entry['id']}"
                        for entry in info['entries']
                        if entry and entry.get('id')
                    ]
                    logger.info(f"Found {len(video_urls)} videos in playlist")
                    return video_urls
            
            return []
            
        except Exception as e:
            logger.error(f"Error extracting playlist videos: {e}")
            return []
    
    def fetch_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Fetch transcript for a YouTube video.
        
        Language priority:
        1. English manual transcript
        2. English auto-generated transcript
        3. First available transcript
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of transcript segments with 'text', 'start', 'duration'
            
        Raises:
            TranscriptsDisabled: If transcripts are disabled for the video
            NoTranscriptFound: If no transcripts are available
            VideoUnavailable: If video is private or deleted
        """
        try:
            # Get available transcripts
            # Note: Using instance method .list() for installed version compatibility
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            # Try to get English manual transcript first
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                logger.info(f"Using manual English transcript for {video_id}")
            except NoTranscriptFound:
                # Try English auto-generated
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                    logger.info(f"Using auto-generated English transcript for {video_id}")
                except NoTranscriptFound:
                    # Get first available transcript
                    transcript = next(iter(transcript_list))
                    logger.info(f"Using {transcript.language} transcript for {video_id}")
            
            # Fetch the transcript data
            transcript_data = transcript.fetch()
            logger.info(f"Fetched {len(transcript_data)} transcript segments")
            
            # Convert to dicts if they are objects (compatibility for v1.2.3)
            cleaned_data = []
            for segment in transcript_data:
                if hasattr(segment, 'text'):
                    cleaned_data.append({
                        'text': segment.text,
                        'start': segment.start,
                        'duration': segment.duration
                    })
                else:
                    cleaned_data.append(segment)
            
            return cleaned_data
            
        except TranscriptsDisabled:
            logger.error(f"Transcripts are disabled for video {video_id}")
            raise
        except NoTranscriptFound:
            logger.error(f"No transcripts found for video {video_id}")
            raise
        except VideoUnavailable:
            logger.error(f"Video {video_id} is unavailable (private or deleted)")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching transcript for {video_id}: {e}")
            raise
    
    def fetch_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Fetch metadata for a YouTube video using yt-dlp.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with metadata (title, channel, duration, etc.)
        """
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                metadata = {
                    'video_id': video_id,
                    'url': url,
                    'source': info.get('title', 'Unknown Title'),
                    'channel': info.get('uploader', 'Unknown Channel'),
                    'duration': info.get('duration', 0),  # In seconds
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', '')[:500],  # First 500 chars
                    'view_count': info.get('view_count', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'document_type': 'youtube'
                }
                
                logger.info(f"Fetched metadata for: {metadata['source']}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error fetching metadata for {video_id}: {e}")
            # Return minimal metadata
            return {
                'video_id': video_id,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'source': f"YouTube Video {video_id}",
                'channel': 'Unknown',
                'duration': 0,
                'document_type': 'youtube'
            }
    
    def fetch_chapters(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Extract chapter information from a YouTube video if available.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of chapters with title, start_time, end_time
        """
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                chapters = info.get('chapters', [])
                
                if chapters:
                    logger.info(f"Found {len(chapters)} chapters for {video_id}")
                    return [
                        {
                            'title': ch.get('title', ''),
                            'start_time': ch.get('start_time', 0),
                            'end_time': ch.get('end_time', 0)
                        }
                        for ch in chapters
                    ]
                
                return []
                
        except Exception as e:
            logger.error(f"Error fetching chapters for {video_id}: {e}")
            return []
    
    def _create_time_chunks(
        self, 
        transcript_data: List[Dict[str, Any]], 
        target_chunk_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Group transcript segments into time-aware chunks.
        
        Args:
            transcript_data: List of transcript segments
            target_chunk_size: Target size in characters
            
        Returns:
            List of chunks with text, start, and end times
        """
        chunks = []
        current_chunk = {
            'text': '',
            'start': 0,
            'end': 0
        }
        
        for i, segment in enumerate(transcript_data):
            text = segment['text']
            start = segment['start']
            duration = segment.get('duration', 0)
            end = start + duration
            
            # Initialize chunk start time
            if not current_chunk['text']:
                current_chunk['start'] = start
            
            # Add segment to current chunk
            current_chunk['text'] += ' ' + text
            current_chunk['end'] = end
            
            # Check if chunk is large enough
            if len(current_chunk['text']) >= target_chunk_size:
                chunks.append({
                    'text': current_chunk['text'].strip(),
                    'start': current_chunk['start'],
                    'end': current_chunk['end']
                })
                current_chunk = {'text': '', 'start': 0, 'end': 0}
        
        # Add remaining chunk
        if current_chunk['text']:
            chunks.append({
                'text': current_chunk['text'].strip(),
                'start': current_chunk['start'],
                'end': current_chunk['end']
            })
        
        return chunks
    
    def load_documents_with_timestamps(
        self, 
        urls: List[str],
        target_chunk_size: int = 1000
    ) -> List[Document]:
        """
        Load YouTube transcripts with timestamp preservation for citations.
        
        Args:
            urls: List of YouTube URLs
            target_chunk_size: Target chunk size in characters
            
        Returns:
            List of LangChain Documents with timestamp metadata
        """
        documents = []
        
        for url in urls:
            try:
                # Parse video ID
                video_id = self._parse_video_id(url)
                if not video_id:
                    logger.warning(f"Could not parse video ID from URL: {url}")
                    continue
                
                # Fetch transcript and metadata
                transcript_data = self.fetch_transcript(video_id)
                metadata = self.fetch_metadata(video_id)
                chapters = self.fetch_chapters(video_id)
                
                # Create time-aware chunks
                time_chunks = self._create_time_chunks(transcript_data, target_chunk_size)
                
                # Create a document for each chunk
                for chunk in time_chunks:
                    # Find chapter for this chunk if available
                    chapter_title = None
                    if chapters:
                        for chapter in chapters:
                            if chapter['start_time'] <= chunk['start'] < chapter['end_time']:
                                chapter_title = chapter['title']
                                break
                    
                    doc_metadata = {
                        **metadata,
                        'start_time': chunk['start'],
                        'end_time': chunk['end'],
                        'timestamp_url': f"{url}&t={int(chunk['start'])}s"
                    }
                    
                    if chapter_title:
                        doc_metadata['chapter'] = chapter_title
                    
                    doc = Document(
                        page_content=chunk['text'],
                        metadata=doc_metadata
                    )
                    documents.append(doc)
                
                logger.info(f"Created {len(time_chunks)} time-aware chunks for: {metadata['source']}")
                
            except Exception as e:
                logger.error(f"Error loading video {url}: {e}")
                continue
        
        return documents
    
    def load_documents(self, urls: List[str]) -> List[Document]:
        """
        Load YouTube video transcripts as LangChain Documents.
        
        This is the main entry point that uses timestamp-aware chunking.
        
        Args:
            urls: List of YouTube URLs (can include playlists if expanded first)
            
        Returns:
            List of LangChain Documents ready for further processing
        """
        return self.load_documents_with_timestamps(urls)

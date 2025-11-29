"""
YouTube Summarizer and Chat Module.

This module provides functionality to:
1. Extract transcripts from YouTube videos.
2. Generate summaries using an LLM.
3. Chat with the video content using an LLM.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from ai_workdesk.rag.youtube_loader import YouTubeTranscriptLoader
from ai_workdesk.tools.llm.ollama_client import OllamaClient
from ai_workdesk.core.config import get_settings

class YouTubeSummarizer:
    """
    Handles YouTube video summarization and chat interactions.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the YouTube Summarizer.
        
        Args:
            llm_client: Optional LLM client instance. If None, uses default OllamaClient.
        """
        self.loader = YouTubeTranscriptLoader()
        self.settings = get_settings()
        
        if llm_client:
            self.llm_client = llm_client
        else:
            # Initialize default Ollama client
            self.llm_client = OllamaClient()

    def get_video_content(self, url: str) -> Dict[str, Any]:
        """
        Get transcript and metadata for a video.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary containing 'text', 'metadata', and 'video_id'
        """
        video_id = self.loader._parse_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")
            
        try:
            # Fetch transcript segments
            transcript_segments = self.loader.fetch_transcript(video_id)
            
            # Combine text
            full_text = " ".join([seg['text'] for seg in transcript_segments])
            
            # Fetch metadata
            metadata = self.loader.fetch_metadata(video_id)
            
            return {
                "video_id": video_id,
                "text": full_text,
                "metadata": metadata,
                "segments": transcript_segments
            }
        except Exception as e:
            logger.error(f"Error getting video content for {url}: {e}")
            raise

    def generate_summary(self, text: str, model: Optional[str] = None) -> str:
        """
        Generate a summary of the provided text.
        
        Args:
            text: Transcript text to summarize
            model: Optional model to use
            
        Returns:
            Summary string
        """
        if not text:
            return "No text provided to summarize."
            
        # Truncate text if too long (simple approach for now)
        # Assuming approx 4 chars per token, 12k chars is ~3k tokens
        # We'll leave room for the prompt and response
        max_chars = 12000 
        if len(text) > max_chars:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars for summary.")
            text = text[:max_chars] + "..."
            
        prompt = f"""
        Please provide a comprehensive summary of the following YouTube video transcript.
        Capture the main points, key arguments, and any important conclusions.
        Format the output with clear headings and bullet points.
        
        TRANSCRIPT:
        {text}
        
        SUMMARY:
        """
        
        try:
            response = self.llm_client.chat(prompt, model=model)
            return response
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

    def chat_with_video(self, text: str, query: str, history: List[Dict], model: Optional[str] = None) -> str:
        """
        Chat with the video content.
        
        Args:
            text: Transcript text
            query: User question
            history: Chat history (list of dicts with 'role' and 'content')
            model: Optional model to use
            
        Returns:
            LLM response
        """
        if not text:
            return "No video content available to chat with."
            
        # Context management
        # We need to include the transcript in the context, but it might be large.
        # For a simple "chat with video", we can put the transcript in the system prompt 
        # or the first user message if it fits.
        
        # Truncate text if needed
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars] + "... (truncated)"
            
        system_prompt = f"""
        You are a helpful assistant that answers questions about a specific YouTube video.
        Use the following transcript to answer the user's questions.
        If the answer is not in the transcript, say so.
        
        VIDEO TRANSCRIPT:
        {text}
        """
        
        # Construct messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history (limit to last few turns to save context)
        for msg in history[-6:]: 
            messages.append(msg)
            
        # Add current query
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.llm_client.chat(messages, model=model)
            return response
        except Exception as e:
            logger.error(f"Error chatting with video: {e}")
            return f"Error: {str(e)}"

"""
YouTube video transcript summarizer.

This module provides automatic summarization of YouTube video transcripts
using LLMs to help users quickly understand video content.
"""

from typing import Optional
from loguru import logger


class YouTubeSummarizer:
    """
    Generates concise summaries of YouTube video transcripts.
    
    Uses LLMs to create brief, informative summaries that capture
    the main topics and key insights from video content.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the YouTube summarizer.
        
        Args:
            llm_client: LLM client with a chat() method
        """
        self.llm = llm_client
    
    def summarize(
        self, 
        transcript: str, 
        video_title: str, 
        max_length: int = 200,
        channel: Optional[str] = None
    ) -> str:
        """
        Generate a concise summary of a YouTube video transcript.
        
        Args:
            transcript: Full video transcript text
            video_title: Video title for context
            max_length: Maximum summary length in words
            channel: Optional channel name for additional context
            
        Returns:
            Brief summary string capturing key points
        """
        try:
            # Truncate transcript if too long (use strategic sampling)
            # Keep first and last portions to capture intro and conclusion
            if len(transcript) > 8000:
                logger.info(f"Truncating long transcript ({len(transcript)} chars)")
                middle_point = len(transcript) // 2
                transcript = (
                    transcript[:4000] + 
                    "\n\n[... middle section omitted ...]\n\n" + 
                    transcript[-4000:]
                )
            
            # Build prompt
            channel_context = f"\nChannel: {channel}" if channel else ""
            
            prompt = f"""Summarize this YouTube video transcript concisely in {max_length} words or less.

Video Title: {video_title}{channel_context}

Transcript:
{transcript}

Provide a clear, informative summary that captures:
1. The main topic or subject
2. Key insights or takeaways
3. Important points discussed

Summary:"""
            
            # Generate summary with low temperature for consistency
            summary = self.llm.chat(prompt)
            
            # Clean up the summary
            summary = summary.strip()
            
            # Ensure it's not too long
            words = summary.split()
            if len(words) > max_length:
                summary = ' '.join(words[:max_length]) + '...'
            
            logger.info(f"Generated summary: {len(summary)} chars, {len(words)} words")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Return a basic fallback summary
            return f"Summary unavailable for: {video_title}"
    
    def batch_summarize(
        self, 
        transcripts: list, 
        video_titles: list,
        channels: Optional[list] = None
    ) -> list:
        """
        Generate summaries for multiple videos in batch.
        
        Args:
            transcripts: List of transcript texts
            video_titles: List of video titles
            channels: Optional list of channel names
            
        Returns:
            List of summary strings
        """
        summaries = []
        channels = channels or [None] * len(transcripts)
        
        for transcript, title, channel in zip(transcripts, video_titles, channels):
            summary = self.summarize(transcript, title, channel=channel)
            summaries.append(summary)
        
        return summaries

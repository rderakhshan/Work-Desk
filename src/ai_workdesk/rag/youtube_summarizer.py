"""
YouTube video transcript summarizer.

This module provides automatic summarization of YouTube video transcripts
using LLMs to create comprehensive, structured summaries following a
scientific paper format.
"""

from typing import Optional
from loguru import logger


class YouTubeSummarizer:
    """
    Generates comprehensive, structured summaries of YouTube video transcripts.
    
    Uses LLMs to create detailed summaries following a scientific paper format
    with introduction, detailed analysis, and conclusion sections.
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
        max_length: int = 800,
        channel: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive, structured summary of a YouTube video transcript.
        
        Args:
            transcript: Full video transcript text
            video_title: Video title for context
            max_length: Maximum summary length in words (default: 800)
            channel: Optional channel name for additional context
            
        Returns:
            Detailed summary string in scientific paper format
        """
        try:
            # Truncate transcript if too long (use strategic sampling)
            # Keep first and last portions to capture intro and conclusion
            if len(transcript) > 12000:
                logger.info(f"Truncating long transcript ({len(transcript)} chars)")
                # For longer transcripts, take more content
                transcript = (
                    transcript[:6000] + 
                    "\n\n[... middle section omitted ...]\n\n" + 
                    transcript[-6000:]
                )
            
            # Build enhanced prompt
            channel_context = f"\nChannel: {channel}" if channel else ""
            
            prompt = f"""You are an expert educational content analyst. Analyze this YouTube video transcript and create a comprehensive, structured summary following a scientific paper format.

Video Title: {video_title}{channel_context}

Transcript:
{transcript}

Create a detailed summary with the following structure:

## 1. INTRODUCTION & OVERVIEW
- Identify the main topic and central theme of the video
- Describe the big picture: What is the overarching concept or problem being addressed?
- List the key tools, techniques, methodologies, or frameworks mentioned
- Explain the context and why this topic matters
- If the video has distinct sections or chapters, list them with their titles

## 2. DETAILED CONTENT ANALYSIS
Break down the video content systematically:

### Main Concepts
- Explain each major concept in simple, clear language (as an experienced teacher would)
- Connect each concept to the big picture established in the introduction
- Highlight relationships between different ideas
- Include specific examples, case studies, or demonstrations mentioned
- Note any formulas, algorithms, or technical details presented

### Key Insights & Takeaways
- What are the most important lessons or discoveries?
- What practical applications or implications are discussed?
- What problems does this solve or what questions does it answer?
- Any warnings, limitations, or caveats mentioned?

### Supporting Details
- Important data, statistics, or research findings
- Expert opinions or quotes
- Real-world examples or use cases
- Tools, resources, or references mentioned

## 3. CONCLUSION & SYNTHESIS
Provide a final summary in TWO formats:

### Narrative Summary (1-2 paragraphs)
Synthesize the entire video into a cohesive story that:
- Recaps the main topic and its significance
- Highlights the most critical insights
- Connects all major points into a unified understanding
- Provides actionable takeaways

### Bullet Point Summary
- 📌 Main Topic: [One sentence]
- 🎯 Key Objective: [What the video aims to achieve]
- 💡 Core Insights: [3-5 most important points]
- 🔧 Tools/Methods: [Techniques or frameworks discussed]
- ✅ Takeaways: [Practical applications or conclusions]
- 🔗 Connections: [How this relates to broader context]

---

IMPORTANT GUIDELINES:
- Use clear, professional language suitable for an educated audience
- Explain technical terms when first introduced
- Maintain logical flow and coherence throughout
- Ensure each detail connects back to the main theme
- Be comprehensive but concise - aim for depth over breadth
- Use analogies or simplified explanations for complex concepts
- Preserve the video's teaching style and pedagogical approach

Summary:"""
            
            # Generate summary with moderate temperature for structured output
            summary = self.llm.chat(prompt)
            
            # Clean up the summary
            summary = summary.strip()
            
            # Log summary statistics
            words = summary.split()
            logger.info(f"Generated summary: {len(summary)} chars, {len(words)} words")
            
            # Note: We don't truncate here as the LLM should follow the structure
            # If it's too long, the LLM settings should be adjusted
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Return a basic fallback summary
            return f"Summary unavailable for: {video_title}\n\nError: {str(e)}"
    
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

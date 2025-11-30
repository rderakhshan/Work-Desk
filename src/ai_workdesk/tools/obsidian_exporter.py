"""
Obsidian Note Exporter for AI Workdesk.

Exports YouTube summaries and chat content as Obsidian-compatible Markdown notes.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ObsidianExporter:
    """Handles exporting content to Obsidian vault as Markdown notes."""
    
    def __init__(self, vault_path: str):
        """
        Initialize the Obsidian exporter.
        
        Args:
            vault_path: Absolute path to the Obsidian vault
        """
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise ValueError(f"Obsidian vault path does not exist: {vault_path}")
        
        # Default subfolder for AI Workdesk notes
        self.workdesk_folder = self.vault_path / "AI Workdesk" / "YouTube"
        self.workdesk_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ObsidianExporter initialized with vault: {vault_path}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for filesystem compatibility.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for all filesystems
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '-')
        
        # Limit length
        max_length = 200
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        return filename.strip()
    
    def export_youtube_note(
        self,
        video_title: str,
        video_url: str,
        channel: str,
        summary: str,
        transcript: Optional[str] = None,
        chat_history: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export a YouTube video summary as an Obsidian note.
        
        Args:
            video_title: Title of the video
            video_url: YouTube URL
            channel: Channel name
            summary: AI-generated summary
            transcript: Optional full transcript
            chat_history: Optional chat Q&A about the video
            metadata: Optional additional metadata
            
        Returns:
            Path to the created note
        """
        try:
            # Create filename
            date_str = datetime.now().strftime("%Y-%m-%d")
            safe_title = self._sanitize_filename(video_title)
            filename = f"{safe_title} - {date_str}.md"
            note_path = self.workdesk_folder / filename
            
            # Build frontmatter
            frontmatter = self._build_frontmatter(
                title=video_title,
                url=video_url,
                channel=channel,
                date_str=date_str,
                metadata=metadata
            )
            
            # Build note content
            content = self._build_note_content(
                title=video_title,
                url=video_url,
                channel=channel,
                summary=summary,
                transcript=transcript,
                chat_history=chat_history
            )
            
            # Write to file
            full_content = f"{frontmatter}\n\n{content}"
            note_path.write_text(full_content, encoding='utf-8')
            
            logger.info(f"Exported Obsidian note to: {note_path}")
            return str(note_path)
            
        except Exception as e:
            logger.error(f"Error exporting Obsidian note: {e}")
            raise
    
    def _build_frontmatter(
        self,
        title: str,
        url: str,
        channel: str,
        date_str: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build YAML frontmatter for the note."""
        frontmatter_lines = [
            "---",
            f"title: \"{title}\"",
            "source: youtube",
            f"url: {url}",
            f"channel: \"{channel}\"",
            f"date: {date_str}",
            "tags:",
            "  - youtube",
            "  - ai-summary",
            "  - ai-workdesk"
        ]
        
        # Add custom metadata if provided
        if metadata:
            for key, value in metadata.items():
                if key not in ['title', 'url', 'channel', 'date']:
                    frontmatter_lines.append(f"{key}: {value}")
        
        frontmatter_lines.append("---")
        return "\n".join(frontmatter_lines)
    
    def _build_note_content(
        self,
        title: str,
        url: str,
        channel: str,
        summary: str,
        transcript: Optional[str] = None,
        chat_history: Optional[str] = None
    ) -> str:
        """Build the main content of the note."""
        content_parts = []
        
        # Title
        content_parts.append(f"# {title}\n")
        
        # Video info
        content_parts.append(f"**Channel:** {channel}")
        content_parts.append(f"**URL:** [{url}]({url})\n")
        
        # Summary section
        content_parts.append("## 📝 Summary\n")
        content_parts.append(f"{summary}\n")
        
        # Chat history (if available)
        if chat_history:
            content_parts.append("## 💬 Chat History\n")
            content_parts.append(f"{chat_history}\n")
        
        # Transcript (collapsible if long)
        if transcript:
            content_parts.append("## 📄 Full Transcript\n")
            if len(transcript) > 1000:
                # Make it collapsible for long transcripts
                content_parts.append("<details>")
                content_parts.append("<summary>Click to expand full transcript</summary>\n")
                content_parts.append(f"{transcript}\n")
                content_parts.append("</details>")
            else:
                content_parts.append(f"{transcript}\n")
        
        return "\n".join(content_parts)

    def export_chat_note(
        self,
        title: str,
        chat_history: str,
        tags: list = None
    ) -> str:
        """
        Export a generic chat session as an Obsidian note.
        
        Args:
            title: Note title
            chat_history: Formatted chat history
            tags: Optional list of tags
            
        Returns:
            Path to the created note
        """
        try:
            # Create filename
            date_str = datetime.now().strftime("%Y-%m-%d")
            safe_title = self._sanitize_filename(title)
            filename = f"{safe_title} - {date_str}.md"
            
            # Save in "Chats" subfolder
            chat_folder = self.workdesk_folder.parent / "Chats"
            chat_folder.mkdir(parents=True, exist_ok=True)
            note_path = chat_folder / filename
            
            # Build frontmatter
            tags = tags or ["ai-chat", "ai-workdesk"]
            frontmatter_lines = [
                "---",
                f"title: \"{title}\"",
                f"date: {date_str}",
                "tags:"
            ]
            for tag in tags:
                frontmatter_lines.append(f"  - {tag}")
            frontmatter_lines.append("---")
            frontmatter = "\n".join(frontmatter_lines)
            
            # Build content
            content = f"# {title}\n\n## 💬 Chat History\n\n{chat_history}"
            
            # Write to file
            full_content = f"{frontmatter}\n\n{content}"
            note_path.write_text(full_content, encoding='utf-8')
            
            logger.info(f"Exported Chat note to: {note_path}")
            return str(note_path)
            
        except Exception as e:
            logger.error(f"Error exporting Chat note: {e}")
            raise

    def export_rag_note(
        self,
        title: str,
        query: str,
        answer: str,
        sources: list = None,
        chat_history: str = None
    ) -> str:
        """
        Export a RAG session as an Obsidian note.
        
        Args:
            title: Note title
            query: The user's query
            answer: The AI's answer
            sources: List of source documents
            chat_history: Optional full chat history
            
        Returns:
            Path to the created note
        """
        try:
            # Create filename
            date_str = datetime.now().strftime("%Y-%m-%d")
            safe_title = self._sanitize_filename(title)
            filename = f"{safe_title} - {date_str}.md"
            
            # Save in "RAG" subfolder
            rag_folder = self.workdesk_folder.parent / "RAG"
            rag_folder.mkdir(parents=True, exist_ok=True)
            note_path = rag_folder / filename
            
            # Build frontmatter
            frontmatter_lines = [
                "---",
                f"title: \"{title}\"",
                f"date: {date_str}",
                "type: rag-session",
                "tags:",
                "  - ai-rag",
                "  - ai-workdesk"
            ]
            frontmatter_lines.append("---")
            frontmatter = "\n".join(frontmatter_lines)
            
            # Build content
            content_parts = [f"# {title}\n"]
            
            content_parts.append("## ❓ Query")
            content_parts.append(f"{query}\n")
            
            content_parts.append("## 💡 Answer")
            content_parts.append(f"{answer}\n")
            
            if sources:
                content_parts.append("## 📚 Sources")
                for source in sources:
                    content_parts.append(f"- {source}")
                content_parts.append("")
            
            if chat_history:
                content_parts.append("## 💬 Full Chat History")
                content_parts.append(f"{chat_history}\n")
            
            # Write to file
            full_content = f"{frontmatter}\n\n" + "\n".join(content_parts)
            note_path.write_text(full_content, encoding='utf-8')
            
            logger.info(f"Exported RAG note to: {note_path}")
            return str(note_path)
            
        except Exception as e:
            logger.error(f"Error exporting RAG note: {e}")
            raise

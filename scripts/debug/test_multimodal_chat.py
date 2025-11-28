"""
Test ChatInterface with MultimodalTextbox for AI Workdesk.

This demonstrates the working microphone recording with minimal code.
"""

import gradio as gr
from pathlib import Path


def chat_multimodal(message, history):
    """
    Handle multimodal chat with text, audio, and files.
    
    Args:
        message: Dict with {"text": str, "files": list} from MultimodalTextbox
        history: List of chat messages
        
    Returns:
        Response string
    """
    # Extract text and files
    user_text = message.get("text", "") if isinstance(message, dict) else str(message)
    user_files = message.get("files", []) if isinstance(message, dict) else []
    
    if not user_text and not user_files:
        return "Please provide a message or upload a file."
    
    # Build response
    response_parts = [f"You said: {user_text}"]
    
    # Process files
    if user_files:
        response_parts.append(f"\n\nYou uploaded {len(user_files)} file(s):")
        for i, file_info in enumerate(user_files, 1):
            if isinstance(file_info, dict):
                file_path = file_info.get("path", "")
                file_name = file_info.get("orig_name", Path(file_path).name)
                file_size = file_info.get("size", 0)
                
                # Check if audio file
                if file_path.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg')):
                    response_parts.append(f"\n{i}. 🎤 Audio: {file_name} ({file_size} bytes)")
                    response_parts.append(f"   ✅ Audio recording detected! In production, this would be transcribed.")
                else:
                    response_parts.append(f"\n{i}. 📄 File: {file_name} ({file_size} bytes)")
    
    return "\n".join(response_parts)


# Create ChatInterface with MultimodalTextbox
demo = gr.ChatInterface(
    fn=chat_multimodal,
    multimodal=True,  # Enable multimodal input
    textbox=gr.MultimodalTextbox(
        sources=["upload", "microphone"],  # Enable both upload AND microphone
        file_types=["audio", "image", "video", "text", ".pdf"],
        file_count="multiple",
        placeholder="Type a message, upload files, or record audio...",
        show_label=False
    ),
    title="🎤 AI Workdesk - Multimodal Chat Test",
    description="Test microphone recording, file uploads, and text input in ONE component!",
    examples=[
        {"text": "Hello! Can you hear me?", "files": []},
        {"text": "What's in this audio?", "files": []},
    ],
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    ),
    css="""
    .contain { max-width: 1200px; margin: auto; }
    """
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7864,  # Fresh port for clean test
        share=False
    )

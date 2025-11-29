
import sys
import os
# from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ai_workdesk.rag.youtube_summarizer import YouTubeSummarizer

def test_summarizer():
    print("Testing YouTube Summarizer...")
    
    # Initialize
    try:
        summarizer = YouTubeSummarizer()
        print("✅ Initialized YouTubeSummarizer")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Test video (Short video: "Me at the zoo" - 18 seconds)
    video_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    
    try:
        print(f"Fetching content for {video_url}...")
        content = summarizer.get_video_content(video_url)
        print(f"✅ Fetched content. Text length: {len(content['text'])}")
        print(f"Metadata: {content['metadata']['source']}")
        
        # Test Summarization
        print("Generating summary...")
        # Mocking the LLM call to avoid actual API usage if needed, but let's try real first if Ollama is running.
        # If Ollama is not running, this might fail.
        # For verification, we can check if we can call it.
        
        summary = summarizer.generate_summary(content['text'])
        print(f"✅ Generated summary: {summary[:100]}...")
        
        # Test Chat
        print("Testing chat...")
        history = []
        response = summarizer.chat_with_video(content['text'], "What is in the video?", history)
        print(f"✅ Chat response: {response}")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_summarizer()

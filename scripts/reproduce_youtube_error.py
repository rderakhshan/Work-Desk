
import logging
from ai_workdesk.rag.youtube_loader import YouTubeTranscriptLoader

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_workdesk.rag.youtube_loader")
logger.setLevel(logging.INFO)

def test_youtube_loader():
    # Test with a video known to have captions (Me at the zoo)
    # Video ID: jNQXAC9IVRw
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    
    print(f"Testing with URL: {test_url}")
    
    loader = YouTubeTranscriptLoader()
    
    try:
        # 1. Test Video ID parsing
        video_id = loader._parse_video_id(test_url)
        print(f"Parsed Video ID: {video_id}")
        
        if not video_id:
            print("❌ Failed to parse Video ID")
            return

        # 2. Test Transcript Fetching directly
        print("Attempting to fetch transcript...")
        try:
            transcript = loader.fetch_transcript(video_id)
            print(f"✅ Transcript fetched successfully! ({len(transcript)} segments)")
            print(f"First segment: {transcript[0]}")
        except Exception as e:
            print(f"❌ Failed to fetch transcript: {e}")
            
        # 3. Test Metadata Fetching
        print("Attempting to fetch metadata...")
        try:
            metadata = loader.fetch_metadata(video_id)
            print(f"✅ Metadata fetched: {metadata.get('source', 'Unknown')}")
        except Exception as e:
            print(f"❌ Failed to fetch metadata: {e}")

        # 4. Test Full Document Loading
        print("Attempting full load_documents...")
        docs = loader.load_documents([test_url])
        if docs:
            print(f"✅ Successfully loaded {len(docs)} document chunks")
        else:
            print("❌ load_documents returned empty list")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_youtube_loader()

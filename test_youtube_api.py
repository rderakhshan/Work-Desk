"""Test YouTube API directly to diagnose the issue."""
from youtube_transcript_api import YouTubeTranscriptApi
from loguru import logger

def test_youtube_api(video_id='j5f2EQf5hkw'):
    """Test the YouTube API with the exact same code as youtube_loader.py"""
    try:
        # This is EXACTLY how it's called in youtube_loader.py
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Try to get English auto-generated
        transcript = transcript_list.find_generated_transcript(['en'])
        logger.info(f"Using auto-generated English transcript for {video_id}")
        
        # Fetch the transcript data
        transcript_data = transcript.fetch()
        logger.info(f"Fetched {len(transcript_data)} transcript segments")
        
        # Convert to dicts
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
        
        print(f"✅ SUCCESS! Got {len(cleaned_data)} segments")
        print(f"First segment: {cleaned_data[0]}")
        return cleaned_data
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_youtube_api()

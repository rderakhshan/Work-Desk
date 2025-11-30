"""Quick test of YouTube API with v0.6.2"""
from youtube_transcript_api import YouTubeTranscriptApi

try:
    # Test with the problematic video
    transcript = YouTubeTranscriptApi.get_transcript('8GGuKOrooJA')
    print(f"✅ SUCCESS! Got {len(transcript)} segments")
    print(f"First segment: {transcript[0]}")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")

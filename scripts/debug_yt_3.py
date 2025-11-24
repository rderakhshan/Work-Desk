
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "jNQXAC9IVRw"

print("Attempt 1: Static call")
try:
    transcript_list = YouTubeTranscriptApi.list(video_id)
    print("Success static call")
except Exception as e:
    print(f"Failed static call: {e}")

print("\nAttempt 2: Instance call")
try:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    print("Success instance call")
except Exception as e:
    print(f"Failed instance call: {e}")


from youtube_transcript_api import YouTubeTranscriptApi
import inspect

print(f"YouTubeTranscriptApi: {YouTubeTranscriptApi}")
print(f"Dir: {dir(YouTubeTranscriptApi)}")

try:
    print(f"list_transcripts: {YouTubeTranscriptApi.list_transcripts}")
except AttributeError as e:
    print(f"Error accessing list_transcripts: {e}")

# Check if it's an instance or class issue (it should be a class with static methods)

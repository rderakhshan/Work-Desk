
from youtube_transcript_api import YouTubeTranscriptApi
import inspect

print("Inspect 'list' method:")
try:
    print(inspect.getdoc(YouTubeTranscriptApi.list))
except Exception as e:
    print(e)

print("\nInspect 'fetch' method:")
try:
    print(inspect.getdoc(YouTubeTranscriptApi.fetch))
except Exception as e:
    print(e)

print("\nInspect 'get_transcript' method:")
try:
    # Check if get_transcript exists (it wasn't in dir, but let's be sure)
    print(YouTubeTranscriptApi.get_transcript)
except AttributeError:
    print("get_transcript does not exist")

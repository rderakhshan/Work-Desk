from typing import List
from datetime import datetime, timedelta
import random
from ..models import DashboardCard, SourceType
from .base import BaseFetcher

class YouTubeFetcher(BaseFetcher):
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def fetch(self, limit: int = 5) -> List[DashboardCard]:
        if self.use_mock:
            return self._fetch_mock(limit)
        return [] 

    def _fetch_mock(self, limit: int) -> List[DashboardCard]:
        mock_videos = [
            {
                "title": "The Future of AI Agents: Beyond LLMs",
                "summary": "Deep dive into autonomous agents, tool use, and the next generation of AI architecture.",
                "urgency": 45,
                "channel": "Andrej Karpathy"
            },
            {
                "title": "Building RAG Systems from Scratch",
                "summary": "A practical guide to implementing Retrieval Augmented Generation with vector databases.",
                "urgency": 60,
                "channel": "AI Engineering"
            },
            {
                "title": "NVIDIA Keynote Highlights",
                "summary": "New GPU architecture revealed, focusing on inference speed and energy efficiency.",
                "urgency": 30,
                "channel": "NVIDIA"
            },
            {
                "title": "Python 3.13: What's New?",
                "summary": "Exploring the JIT compiler and removal of the GIL in the latest Python release.",
                "urgency": 20,
                "channel": "Real Python"
            }
        ]
        
        cards = []
        for i in range(min(limit, len(mock_videos))):
            data = mock_videos[i]
            cards.append(DashboardCard(
                title=data["title"],
                summary=data["summary"],
                source_type=SourceType.YOUTUBE,
                source_link="https://youtube.com",
                urgency_score=data["urgency"],
                timestamp=datetime.now() - timedelta(hours=random.randint(0, 48)),
                metadata={"channel": data["channel"]}
            ))
        return cards

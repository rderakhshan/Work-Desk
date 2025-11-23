from typing import List
from datetime import datetime
from ..models import DashboardCard, SourceType
from .base import BaseFetcher

class TrendsFetcher(BaseFetcher):
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def fetch(self, limit: int = 5) -> List[DashboardCard]:
        if self.use_mock:
            return self._fetch_mock(limit)
        return []

    def _fetch_mock(self, limit: int) -> List[DashboardCard]:
        mock_trends = [
            {
                "title": "#AIRegulation",
                "summary": "Global discussions heating up regarding new EU AI Act and its impact on open source models.",
                "urgency": 75
            },
            {
                "title": "React Server Components",
                "summary": "Trending in Web Dev: The shift towards server-first rendering architectures.",
                "urgency": 40
            },
            {
                "title": "Quantum Supremacy",
                "summary": "New breakthrough claims in error-corrected quantum computing.",
                "urgency": 55
            },
            {
                "title": "SpaceX Starship",
                "summary": "Upcoming orbital flight test schedule and mission objectives.",
                "urgency": 30
            }
        ]
        
        cards = []
        for i in range(min(limit, len(mock_trends))):
            data = mock_trends[i]
            cards.append(DashboardCard(
                title=data["title"],
                summary=data["summary"],
                source_type=SourceType.TRENDS,
                source_link="https://trends.google.com",
                urgency_score=data["urgency"],
                timestamp=datetime.now(),
                metadata={"region": "Global"}
            ))
        return cards

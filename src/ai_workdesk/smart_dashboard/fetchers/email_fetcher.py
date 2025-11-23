from typing import List
from datetime import datetime, timedelta
import random
from ..models import DashboardCard, SourceType
from .base import BaseFetcher

class EmailFetcher(BaseFetcher):
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def fetch(self, limit: int = 5) -> List[DashboardCard]:
        if self.use_mock:
            return self._fetch_mock(limit)
        return [] # Real implementation would go here

    def _fetch_mock(self, limit: int) -> List[DashboardCard]:
        mock_emails = [
            {
                "title": "Project Deadline Update",
                "summary": "The deadline for the Q4 report has been moved to next Friday. Please update your slides.",
                "urgency": 85
            },
            {
                "title": "Lunch Plans?",
                "summary": "Hey, are we still on for sushi today at 12:30?",
                "urgency": 20
            },
            {
                "title": "Server Alert: High CPU Usage",
                "summary": "Server 'prod-db-01' is experiencing high CPU load (95%) for the last 10 minutes.",
                "urgency": 95
            },
            {
                "title": "Newsletter: AI Weekly",
                "summary": "Top stories: GPT-5 rumors, new vision models from Google, and more.",
                "urgency": 10
            },
            {
                "title": "Invoice #10234 Overdue",
                "summary": "Your payment for invoice #10234 was due yesterday. Please remit payment immediately.",
                "urgency": 90
            }
        ]
        
        cards = []
        for i in range(min(limit, len(mock_emails))):
            data = mock_emails[i]
            cards.append(DashboardCard(
                title=data["title"],
                summary=data["summary"],
                source_type=SourceType.EMAIL,
                source_link="https://gmail.com",
                urgency_score=data["urgency"],
                timestamp=datetime.now() - timedelta(hours=random.randint(0, 24)),
                metadata={"sender": "mock@example.com"}
            ))
        return cards

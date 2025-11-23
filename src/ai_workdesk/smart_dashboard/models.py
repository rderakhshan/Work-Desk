from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class SourceType(str, Enum):
    EMAIL = "email"
    RSS = "rss"
    YOUTUBE = "youtube"
    TRENDS = "trends"

class DashboardCard(BaseModel):
    """Represents a single card on the Smart Dashboard."""
    title: str
    summary: str
    source_type: SourceType
    source_link: Optional[str] = None
    urgency_score: int = Field(default=0, ge=0, le=100, description="0-100 score indicating importance")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict, description="Extra data like channel name, sender, etc.")

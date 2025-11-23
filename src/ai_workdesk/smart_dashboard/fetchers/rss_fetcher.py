import feedparser
from typing import List
from datetime import datetime
from time import mktime
from ..models import DashboardCard, SourceType
from .base import BaseFetcher

class RSSFetcher(BaseFetcher):
    def __init__(self, feed_urls: List[str] = None):
        self.feed_urls = feed_urls or [
            "http://feeds.bbci.co.uk/news/technology/rss.xml",
            "https://techcrunch.com/feed/",
            "https://www.theverge.com/rss/index.xml"
        ]

    def fetch(self, limit: int = 5) -> List[DashboardCard]:
        cards = []
        for url in self.feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit]:
                    # Parse timestamp
                    dt = datetime.now()
                    if hasattr(entry, 'published_parsed'):
                        dt = datetime.fromtimestamp(mktime(entry.published_parsed))
                    
                    cards.append(DashboardCard(
                        title=entry.title,
                        summary=entry.summary[:200] + "..." if hasattr(entry, 'summary') else "",
                        source_type=SourceType.RSS,
                        source_link=entry.link,
                        urgency_score=10, # Default low urgency for news
                        timestamp=dt,
                        metadata={"feed": feed.feed.title if hasattr(feed, 'feed') else url}
                    ))
            except Exception as e:
                print(f"Error fetching RSS {url}: {e}")
        
        return cards

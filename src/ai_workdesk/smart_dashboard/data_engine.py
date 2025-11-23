from typing import List
import logging
from .models import DashboardCard
from .fetchers.email_fetcher import EmailFetcher
from .fetchers.rss_fetcher import RSSFetcher
from .fetchers.youtube_fetcher import YouTubeFetcher
from .fetchers.trends_fetcher import TrendsFetcher
from .rag_engine import DashboardRAG

logger = logging.getLogger(__name__)

class DataEngine:
    def __init__(self):
        self.fetchers: List[BaseFetcher] = []
        self.fetchers.append(EmailFetcher(use_mock=True))
        self.fetchers.append(RSSFetcher())
        self.fetchers.append(YouTubeFetcher(use_mock=True))
        self.fetchers.append(TrendsFetcher(use_mock=True))
        
        # Initialize RAG engine for semantic search
        self.rag_engine = DashboardRAG()

    def refresh_all(self) -> List[DashboardCard]:
        """Fetches data from all registered sources and embeds them for RAG."""
        all_cards = []
        for fetcher in self.fetchers:
            try:
                cards = fetcher.fetch()
                all_cards.extend(cards)
            except Exception as e:
                logger.error(f"Error fetching from {fetcher.__class__.__name__}: {e}")
        
        # Sort by urgency (descending) and then timestamp (descending)
        all_cards.sort(key=lambda x: (x.urgency_score, x.timestamp), reverse=True)
        
        # Auto-embed cards for RAG
        try:
            self.rag_engine.embed_cards(all_cards)
            logger.info(f"Embedded {len(all_cards)} cards for RAG")
        except Exception as e:
            logger.error(f"Error embedding cards for RAG: {e}")
        
        return all_cards

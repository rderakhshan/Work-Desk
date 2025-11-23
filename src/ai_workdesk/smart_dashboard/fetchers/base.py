from abc import ABC, abstractmethod
from typing import List
from ..models import DashboardCard

class BaseFetcher(ABC):
    """Abstract base class for all data fetchers."""
    
    @abstractmethod
    def fetch(self, limit: int = 5) -> List[DashboardCard]:
        """
        Fetch data from the source and return a list of DashboardCards.
        
        Args:
            limit: Maximum number of items to fetch.
            
        Returns:
            List[DashboardCard]: The fetched and processed cards.
        """
        pass

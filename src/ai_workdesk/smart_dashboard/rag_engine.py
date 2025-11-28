from typing import List, Optional
import logging
from datetime import datetime
from .models import DashboardCard, SourceType
from ..rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class DashboardRAG:
    """RAG engine for Smart Homepage dashboard items.
    
    Handles vector embeddings and semantic retrieval of dashboard cards
    using ChromaDB for scalable, context-aware chat.
    """
    
    def __init__(self, collection_name: str = "dashboard_items"):
        """Initialize the RAG engine with a dedicated ChromaDB collection."""
        self.collection_name = collection_name
        
        # Initialize VectorStoreManager, which will use settings from config
        self.vector_store = VectorStoreManager()
        
        # Override collection name for dashboard items
        self.vector_store.collection_name = collection_name
        
        # Reinitialize the vector store with new collection name
        from langchain_chroma import Chroma
        self.vector_store.vector_store = Chroma(
            client=self.vector_store.client,
            collection_name=self.collection_name,
            embedding_function=self.vector_store.embedding_function,
        )
        
        logger.info(f"DashboardRAG initialized with collection: {collection_name}")
    
    def embed_cards(self, cards: List[DashboardCard]) -> None:
        """Embed dashboard cards into the vector store.
        
        Args:
            cards: List of dashboard cards to embed
        """
        if not cards:
            logger.warning("No cards to embed")
            return
        
        try:
            # Clear existing embeddings to avoid duplicates
            self.clear_collection()
            
            # Prepare Document objects for embedding
            from langchain_core.documents import Document
            documents = []
            
            for i, card in enumerate(cards):
                # Combine title and summary for rich semantic content
                doc_text = f"{card.title}\n{card.summary}"
                
                # Create Document with metadata
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        "title": card.title,
                        "summary": card.summary,
                        "source_type": card.source_type.value,
                        "source_link": card.source_link,
                        "urgency_score": card.urgency_score,
                        "timestamp": card.timestamp.isoformat(),
                        "card_id": f"card_{i}_{card.timestamp.timestamp()}"
                    }
                )
                documents.append(doc)
            
            # Add to vector store using the correct method signature
            self.vector_store.add_documents(documents)
            
            logger.info(f"Successfully embedded {len(cards)} dashboard cards")
            
        except Exception as e:
            logger.error(f"Error embedding cards: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[DashboardCard]:
        """Perform semantic search to find relevant dashboard items.
        
        Args:
            query: User's question or search query
            top_k: Number of most relevant items to retrieve
            
        Returns:
            List of most relevant DashboardCard objects
        """
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k
            )
            
            if not results:
                logger.warning(f"No results found for query: {query}")
                return []
            
            # Convert results back to DashboardCard objects
            cards = []
            for doc in results:
                metadata = doc.metadata
                
                # Reconstruct DashboardCard from metadata
                card = DashboardCard(
                    title=metadata.get("title", ""),
                    summary=metadata.get("summary", ""),
                    source_type=SourceType(metadata.get("source_type", "rss")),
                    source_link=metadata.get("source_link", ""),
                    urgency_score=metadata.get("urgency_score", 0),
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
                )
                cards.append(card)
            
            logger.info(f"Retrieved {len(cards)} relevant cards for query: {query[:50]}...")
            return cards
            
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []
    
    def clear_collection(self) -> None:
        """Clear all embeddings from the collection."""
        try:
            # Delete and recreate collection for fresh start
            self.vector_store.client.delete_collection(name=self.collection_name)
            
            # Reinitialize vector store
            from langchain_chroma import Chroma
            self.vector_store.vector_store = Chroma(
                client=self.vector_store.client,
                collection_name=self.collection_name,
                embedding_function=self.vector_store.embedding_function,
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Error clearing collection: {e}")

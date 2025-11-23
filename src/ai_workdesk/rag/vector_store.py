import os
from typing import List, Optional
from loguru import logger
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma

class VectorStoreManager:
    """Manages interactions with the Vector Database (ChromaDB) and supports multiple embedding providers."""

    def __init__(self, persist_directory: str = "./chroma_db", embedding_provider: str = "huggingface", embedding_model: Optional[str] = None):
        """Initialize VectorStoreManager with configurable embedding provider.

        Args:
            persist_directory: Directory for ChromaDB persistence.
            embedding_provider: One of "huggingface", "ollama", "openai".
            embedding_model: Specific model name; if None, defaults from Settings are used.
        """
        from ai_workdesk.core.config import get_settings
        settings = get_settings()

        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "ai_workdesk_collection"

        # Determine embedding function based on provider
        if embedding_provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            model_name = embedding_model or settings.ollama_embedding_model
            self.embedding_function = OllamaEmbeddings(model=model_name, base_url=settings.ollama_base_url)
        elif embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            model_name = embedding_model or settings.default_embedding_model
            self.embedding_function = OpenAIEmbeddings(model=model_name)
        else:  # default to huggingface
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = embedding_model or "all-MiniLM-L6-v2"
            self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)

        # Initialize LangChain wrapper with the chosen embedding function
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        logger.info(f"VectorStoreManager initialized at {persist_directory} with provider {embedding_provider}")

    def add_documents(self, chunks: List[Document]) -> bool:
        """Add document chunks to the vector store.

        Args:
            chunks: List of Document chunks.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not chunks:
                return False
            self.vector_store.add_documents(documents=chunks)
            # Log collection stats
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            logger.info(f"Added {len(chunks)} chunks. Total documents in collection: {count}")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False

    def get_stats(self) -> dict:
        """Get current statistics of the vector store."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            return {
                "status": "Ready",
                "total_chunks": count,
                "vector_db": "ChromaDB",
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "status": "Error",
                "error": str(e),
            }

    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Document]:
        """Search for similar documents in the vector store.

        Args:
            query: The search query.
            k: Number of results to return.
            score_threshold: Minimum similarity score (0.0-1.0).

        Returns:
            List of relevant Document chunks.
        """
        try:
            # Check collection stats first
            collection = self.client.get_collection(self.collection_name)
            total_docs = collection.count()
            logger.info(f"Searching in collection with {total_docs} total documents")
            
            if total_docs == 0:
                logger.warning("Collection is EMPTY! No documents to search.")
                return []
            
            # Use similarity_search_with_score to get scores
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            # ChromaDB returns L2 distance (lower is better: 0 = perfect match)
            # Convert similarity threshold (0-1, higher is better) to max distance
            # For threshold 0.7, we want distance <= 0.3
            max_distance = 1.0 - score_threshold
            
            filtered_docs = [
                doc for doc, distance in results_with_scores 
                if distance <= max_distance
            ]
            
            logger.info(f"Retrieved {len(filtered_docs)}/{len(results_with_scores)} documents (threshold: {score_threshold}, max_distance: {max_distance:.2f})")
            if len(filtered_docs) == 0 and len(results_with_scores) > 0:
                distances = [f"{d:.3f}" for _, d in results_with_scores[:3]]
                logger.warning(f"All results filtered out! Top 3 distances: {distances}. Try lowering similarity_threshold (current: {score_threshold})")
            return filtered_docs
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

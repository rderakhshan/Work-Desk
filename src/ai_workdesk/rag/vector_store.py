import os
from typing import List, Optional
from loguru import logger
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

class VectorStoreManager:
    """Manages interactions with the Vector Database (ChromaDB)."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "ai_workdesk_collection"
        
        # Initialize LangChain wrapper
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        logger.info(f"VectorStoreManager initialized at {persist_directory}")

    def add_documents(self, chunks: List[Document]) -> bool:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of Document chunks.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if not chunks:
                return False
                
            self.vector_store.add_documents(documents=chunks)
            logger.info(f"Added {len(chunks)} chunks to vector store")
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
                "vector_db": "ChromaDB"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "status": "Error",
                "error": str(e)
            }

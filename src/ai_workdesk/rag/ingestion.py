import os
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles loading and chunking of documents."""

    def __init__(self):
        self.supported_extensions = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
        }

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from the given file paths.
        
        Args:
            file_paths: List of absolute paths to files.
            
        Returns:
            List of LangChain Documents.
        """
        documents = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            ext = path.suffix.lower()
            if ext not in self.supported_extensions:
                logger.warning(f"Unsupported file type: {ext} for file {file_path}")
                continue

            try:
                loader_cls = self.supported_extensions[ext]
                loader = loader_cls(str(path))
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} pages/documents from {path.name}")
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return documents

    def chunk_documents(
        self, 
        documents: List[Document], 
        chunk_size: int = 512, 
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split.
            chunk_size: Size of each chunk in tokens (approx).
            chunk_overlap: Overlap between chunks.
            
        Returns:
            List of chunked Documents.
        """
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

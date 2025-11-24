import os
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader,  # For text-based PDFs (fallback strategy)
    UnstructuredPDFLoader,  # OCR-enabled PDF loader for scanned PDFs
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    RecursiveUrlLoader,
    UnstructuredImageLoader  # OCR-enabled image loader
)
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles loading and chunking of documents."""

    def __init__(self):
        self.supported_extensions = {
            ".txt": TextLoader,
            ".pdf": UnstructuredPDFLoader,  # OCR-enabled for scanned PDFs
            ".md": UnstructuredMarkdownLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".json": JSONLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
            # Image formats with OCR support
            ".png": UnstructuredImageLoader,
            ".jpg": UnstructuredImageLoader,
            ".jpeg": UnstructuredImageLoader,
            ".tiff": UnstructuredImageLoader,
            ".tif": UnstructuredImageLoader,
            ".bmp": UnstructuredImageLoader,
            ".gif": UnstructuredImageLoader,
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
                # Special handling for PDFs: try PyPDFLoader first, then UnstructuredPDFLoader
                if ext == ".pdf":
                    docs = self._load_pdf_with_fallback(str(path))
                else:
                    loader_cls = self.supported_extensions[ext]
                    loader = loader_cls(str(path))
                    docs = loader.load()
                
                logger.info(f"Loaded {len(docs)} pages/documents from {path.name}")
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        return documents
    
    def _load_pdf_with_fallback(self, file_path: str) -> List[Document]:
        """
        Load PDF with fallback strategy: try PyPDFLoader first, then UnstructuredPDFLoader.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of LangChain Documents
        """
        from langchain_community.document_loaders import PyPDFLoader
        
        # Try PyPDFLoader first (faster, works for text-based PDFs)
        try:
            logger.info(f"Attempting to load PDF with PyPDFLoader: {file_path}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Check if we got meaningful content
            if docs and any(len(doc.page_content.strip()) > 50 for doc in docs):
                logger.info(f"Successfully loaded PDF with PyPDFLoader: {len(docs)} pages")
                return docs
            else:
                logger.warning("PyPDFLoader returned no meaningful content, trying UnstructuredPDFLoader...")
        except Exception as e:
            logger.warning(f"PyPDFLoader failed: {e}, trying UnstructuredPDFLoader...")
        
        # Fallback to UnstructuredPDFLoader (for scanned PDFs with OCR)
        try:
            logger.info(f"Attempting to load PDF with UnstructuredPDFLoader: {file_path}")
            loader = UnstructuredPDFLoader(file_path)
            docs = loader.load()
            
            # Validate content
            if not docs or not any(len(doc.page_content.strip()) > 10 for doc in docs):
                raise ValueError(
                    "PDF loaded but contains no text. If this is a scanned document, "
                    "you need to install Tesseract OCR. "
                    "See: https://github.com/tesseract-ocr/tesseract"
                )
                
            logger.info(f"Successfully loaded PDF with UnstructuredPDFLoader: {len(docs)} pages")
            return docs
        except Exception as e:
            logger.error(f"UnstructuredPDFLoader failed: {e}")
            # If it was our validation error, re-raise it
            if "Tesseract OCR" in str(e):
                raise e
            raise  # Re-raise other errors to be caught by the caller

    def load_web_documents(self, url: str, max_depth: int = 2) -> List[Document]:
        """
        Load documents from a web URL with recursive crawling.
        
        Args:
            url: The starting URL.
            max_depth: Maximum depth for recursive crawling.
            
        Returns:
            List of LangChain Documents.
        """
        try:
            logger.info(f"Crawling {url} with depth {max_depth}...")
            
            def simple_extractor(html):
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text().strip()

            loader = RecursiveUrlLoader(
                url=url,
                max_depth=max_depth,
                extractor=simple_extractor,
                prevent_outside=True,
                use_async=True,
                timeout=10,
            )
            
            docs = loader.load()
            logger.info(f"Crawled {len(docs)} pages from {url}")
            return docs
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return []

    def load_youtube_documents(self, video_urls: List[str]) -> List[Document]:
        """
        Load documents from YouTube video URLs.
        
        This method fetches YouTube video transcripts with timestamp preservation
        for precise citations in RAG responses.
        
        Args:
            video_urls: List of YouTube URLs (individual videos or playlists)
            
        Returns:
            List of LangChain Documents with transcript content and rich metadata
        """
        try:
            from .youtube_loader import YouTubeTranscriptLoader
            
            loader = YouTubeTranscriptLoader()
            documents = loader.load_documents(video_urls)
            
            logger.info(f"Loaded {len(documents)} YouTube document chunks from {len(video_urls)} video(s)")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading YouTube documents: {e}")
            return []

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

    def chunk_documents_semantic(
        self, 
        documents: List[Document],
        breakpoint_threshold: float = 0.5
    ) -> List[Document]:
        """
        Split documents using semantic chunking (topic-based breakpoints).
        
        Args:
            documents: List of documents to split.
            breakpoint_threshold: Similarity threshold for detecting topic changes (0-1).
            
        Returns:
            List of semantically chunked Documents.
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_huggingface import HuggingFaceEmbeddings
            
            if not documents:
                return []
            
            # Use lightweight embedding model for chunking
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            text_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=int(breakpoint_threshold * 100)
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Semantically split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}. Falling back to fixed-size chunking.")
            return self.chunk_documents(documents)

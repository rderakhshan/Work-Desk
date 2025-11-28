"""
Advanced Features Module: OCR, RAGAS, Multimodal, Data Cleaning
"""

import os
from typing import List, Dict, Tuple, Optional
from loguru import logger
import numpy as np

# OCR Imports
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR dependencies not available")

# RAGAS Imports
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available")

# Multimodal Imports
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available")

# PII Detection Imports
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PII_AVAILABLE = True
except ImportError:
    PII_AVAILABLE = False
    logger.warning("Presidio not available")


class OCRProcessor:
    """OCR processing for scanned documents and images."""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """Initialize OCR processor."""
        if not OCR_AVAILABLE:
            raise ImportError("OCR dependencies not installed. Install: pip install pytesseract pillow pdf2image")
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        logger.info("OCR Processor initialized")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image file."""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            logger.info(f"Extracted {len(text)} characters from {image_path}")
            return text
        except Exception as e:
            logger.error(f"OCR error for {image_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a scanned PDF."""
        try:
            images = convert_from_path(pdf_path)
            text_parts = []
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                text_parts.append(text)
                logger.info(f"Processed page {i+1}/{len(images)}")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
        except Exception as e:
            logger.error(f"PDF OCR error for {pdf_path}: {e}")
            return ""


class RAGEvaluator:
    """RAG evaluation using RAGAS metrics."""
    
    def __init__(self):
        """Initialize RAGAS evaluator."""
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS not installed. Install: pip install ragas")
        
        logger.info("RAGAS Evaluator initialized")
    
    def evaluate_rag(self, questions: List[str], answers: List[str], 
                     contexts: List[List[str]], ground_truths: List[str]) -> Dict:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts (list of strings for each question)
            ground_truths: List of ground truth answers
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            from datasets import Dataset
            
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths
            }
            
            dataset = Dataset.from_dict(data)
            
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )
            
            logger.info(f"RAGAS evaluation complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation error: {e}")
            return {"error": str(e)}


class MultimodalEmbedder:
    """Multimodal embeddings using CLIP."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP model."""
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not installed. Install: pip install transformers")
        
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        logger.info("CLIP model loaded")
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate embedding for an image."""
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            embedding = image_features.detach().numpy()[0]
            logger.info(f"Generated embedding for {image_path}")
            return embedding
        except Exception as e:
            logger.error(f"Image embedding error: {e}")
            return np.array([])
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using CLIP."""
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            text_features = self.model.get_text_features(**inputs)
            embedding = text_features.detach().numpy()[0]
            return embedding
        except Exception as e:
            logger.error(f"Text embedding error: {e}")
            return np.array([])


class DataCleaner:
    """Data cleaning utilities: PII redaction and deduplication."""
    
    def __init__(self):
        """Initialize data cleaner."""
        if PII_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            logger.info("PII detection enabled")
        else:
            self.analyzer = None
            self.anonymizer = None
            logger.warning("PII detection not available")
    
    def detect_pii(self, text: str) -> List[Dict]:
        """Detect PII in text."""
        if not self.analyzer:
            return []
        
        try:
            results = self.analyzer.analyze(text=text, language='en')
            pii_entities = [
                {
                    "type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": r.score,
                    "text": text[r.start:r.end]
                }
                for r in results
            ]
            logger.info(f"Detected {len(pii_entities)} PII entities")
            return pii_entities
        except Exception as e:
            logger.error(f"PII detection error: {e}")
            return []
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        if not self.anonymizer or not self.analyzer:
            return text
        
        try:
            results = self.analyzer.analyze(text=text, language='en')
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
            logger.info("PII redacted")
            return anonymized.text
        except Exception as e:
            logger.error(f"PII redaction error: {e}")
            return text
    
    def deduplicate_documents(self, documents: List[str], threshold: float = 0.95) -> List[str]:
        """
        Remove duplicate documents based on similarity.
        
        Args:
            documents: List of document texts
            threshold: Similarity threshold for considering duplicates
            
        Returns:
            List of unique documents
        """
        if len(documents) <= 1:
            return documents
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Vectorize documents
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Compute similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            unique_indices = []
            for i in range(len(documents)):
                is_duplicate = False
                for j in unique_indices:
                    if similarity_matrix[i][j] >= threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_indices.append(i)
            
            unique_docs = [documents[i] for i in unique_indices]
            logger.info(f"Deduplication: {len(documents)} -> {len(unique_docs)} documents")
            return unique_docs
            
        except Exception as e:
            logger.error(f"Deduplication error: {e}")
            return documents

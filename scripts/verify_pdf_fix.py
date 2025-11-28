import sys
import os
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# Add src to path
sys.path.append(os.path.abspath("src"))

from ai_workdesk.rag.ingestion import DocumentProcessor

def test_pdf_validation():
    print("Testing PDF validation logic...")
    
    processor = DocumentProcessor()
    
    # Mock UnstructuredPDFLoader to return empty content
    with patch("ai_workdesk.rag.ingestion.UnstructuredPDFLoader") as MockLoader:
        # Case 1: Empty document list
        mock_instance = MockLoader.return_value
        mock_instance.load.return_value = []
        
        try:
            print("Test Case 1: Empty document list")
            processor._load_pdf_with_fallback("dummy.pdf")
            print("FAILED: Should have raised ValueError")
        except ValueError as e:
            if "Tesseract OCR" in str(e):
                print("PASSED: Caught expected error for empty list")
            else:
                print(f"FAILED: Caught wrong error: {e}")
        except Exception as e:
            print(f"FAILED: Caught unexpected exception: {e}")

        # Case 2: Document with empty content
        mock_instance.load.return_value = [Document(page_content="   \n  ", metadata={})]
        
        try:
            print("\nTest Case 2: Document with whitespace content")
            processor._load_pdf_with_fallback("dummy.pdf")
            print("FAILED: Should have raised ValueError")
        except ValueError as e:
            if "Tesseract OCR" in str(e):
                print("PASSED: Caught expected error for whitespace content")
            else:
                print(f"FAILED: Caught wrong error: {e}")
        except Exception as e:
            print(f"FAILED: Caught unexpected exception: {e}")

        # Case 3: Valid content
        mock_instance.load.return_value = [Document(page_content="This is valid content with enough length.", metadata={})]
        
        try:
            print("\nTest Case 3: Valid content")
            docs = processor._load_pdf_with_fallback("dummy.pdf")
            if len(docs) == 1:
                print("PASSED: Successfully loaded valid content")
            else:
                print("FAILED: Did not return documents")
        except Exception as e:
            print(f"FAILED: Should not have raised exception: {e}")

if __name__ == "__main__":
    test_pdf_validation()

import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

try:
    from ai_workdesk.rag.advanced_features import OCR_AVAILABLE, OCRProcessor
    print(f"OCR_AVAILABLE: {OCR_AVAILABLE}")
    
    if OCR_AVAILABLE:
        try:
            import pytesseract
            print(f"Tesseract cmd: {pytesseract.pytesseract.tesseract_cmd}")
            # Check if tesseract is actually installed/callable
            print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print(f"Error checking tesseract: {e}")
            
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

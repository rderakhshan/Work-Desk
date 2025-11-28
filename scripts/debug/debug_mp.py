import sys
print(f"Python version: {sys.version}")

try:
    import multiprocess
    print(f"multiprocess version: {multiprocess.__version__}")
    print(f"multiprocess file: {multiprocess.__file__}")
except ImportError:
    print("multiprocess not installed")

try:
    from presidio_analyzer import AnalyzerEngine
    print("Presidio Analyzer imported")
    engine = AnalyzerEngine()
    print("Presidio Analyzer initialized")
except Exception as e:
    print(f"Presidio error: {e}")

import multiprocessing
print(f"multiprocessing file: {multiprocessing.__file__}")

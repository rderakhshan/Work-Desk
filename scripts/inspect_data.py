
import sys
import os
import spacy

# Add src to path
sys.path.append(os.path.abspath("src"))

from ai_workdesk.core.config import get_settings
from ai_workdesk.rag.vector_store import VectorStoreManager
from ai_workdesk.rag.graph_rag import GraphRAG

def inspect_data():
    print("--- Inspecting Data ---")
    settings = get_settings()
    print(f"Persist Directory: {settings.chroma_persist_directory}")
    print(f"Collection Name: {settings.chroma_collection_name}")
    
    vsm = VectorStoreManager()
    docs = vsm.get_all_documents()
    
    print(f"Total Documents Found: {len(docs)}")
    
    if not docs:
        print("WARNING: No documents found in vector store.")
        return

    print("\n--- Sample Document Content ---")
    for i, doc in enumerate(docs[:3]):
        content_preview = doc.page_content[:200].replace('\n', ' ')
        print(f"Doc {i+1} ({len(doc.page_content)} chars): {content_preview}...")
        print(f"Metadata: {doc.metadata}")

    print("\n--- Testing Entity Extraction ---")
    graph_rag = GraphRAG()
    
    # Test on the first document
    if docs:
        text = docs[0].page_content
        entities = graph_rag.extract_entities(text)
        print(f"Entities found in Doc 1: {len(entities)}")
        print(f"Entities: {entities}")
        
        # Test on a known string to verify model
        test_str = "Apple is looking at buying U.K. startup for $1 billion. Elon Musk is the CEO of Tesla."
        print(f"\nControl Test ('{test_str}'):")
        control_ents = graph_rag.extract_entities(test_str)
        print(f"Control Entities: {control_ents}")

if __name__ == "__main__":
    inspect_data()

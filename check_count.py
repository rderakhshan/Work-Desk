
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from ai_workdesk.rag.vector_store import VectorStoreManager

def check_count():
    try:
        vsm = VectorStoreManager()
        docs = vsm.get_all_documents()
        print(f"COUNT:{len(docs)}")
        
        print("\n--- SOURCES ---")
        sources = set()
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            sources.add(source)
            # Print preview if it's a file
            if "http" not in source:
                 print(f"FILE: {source} (Len: {len(doc.page_content)})")
            else:
                 print(f"URL: {source} (Len: {len(doc.page_content)})")
        
        print(f"\nUNIQUE SOURCES: {sources}")

    except Exception as e:
        print(f"ERROR:{e}")

if __name__ == "__main__":
    check_count()

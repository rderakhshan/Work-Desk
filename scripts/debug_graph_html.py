
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from ai_workdesk.rag.vector_store import VectorStoreManager
from ai_workdesk.rag.graph_rag import GraphRAG

def debug_graph():
    print("--- Generating Debug Graph HTML ---")
    
    vector_store = VectorStoreManager()
    graph_rag = GraphRAG()
    
    # Fetch all docs
    all_docs = vector_store.get_all_documents()
    print(f"Retrieved {len(all_docs)} docs.")
    
    if not all_docs:
        print("No docs found. Please ingest sample data first.")
        return

    # Build graph
    graph_rag.build_graph([doc.page_content for doc in all_docs], clear=True)
    
    # Save to specific file
    output_path = "debug_graph.html"
    graph_rag.visualize_graph(output_path)
    
    print(f"Graph saved to {output_path}")
    
    # Read and print first 50 lines
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("\n--- HTML PREVIEW ---")
        for line in lines[:50]:
            print(line.strip())

if __name__ == "__main__":
    debug_graph()

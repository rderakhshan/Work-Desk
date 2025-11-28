
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from ai_workdesk.rag.ingestion import DocumentProcessor
from ai_workdesk.rag.vector_store import VectorStoreManager
from ai_workdesk.rag.graph_rag import GraphRAG
from ai_workdesk.rag.metadata_store import MetadataStore

def ingest_and_verify():
    print("--- Ingesting Sample Data ---")
    
    # Initialize components
    doc_processor = DocumentProcessor()
    vector_store = VectorStoreManager()
    metadata_store = MetadataStore()
    graph_rag = GraphRAG()
    
    # File to ingest
    file_path = os.path.abspath("data/sample_ai_history.txt")
    print(f"Processing file: {file_path}")
    
    # Process
    documents = doc_processor.process_files([file_path])
    print(f"Chunks generated: {len(documents)}")
    
    if not documents:
        print("ERROR: No documents processed.")
        return
        
    # Add to vector store
    vector_store.add_documents(documents)
    print("Added to vector store.")
    
    # Add to metadata store
    metadata_store.add_entry(
        filename="sample_ai_history.txt",
        size=os.path.getsize(file_path),
        doc_type="txt"
    )
    print("Added to metadata store.")
    
    # Verify Graph
    print("\n--- Verifying Graph ---")
    
    # Fetch all docs
    all_docs = vector_store.get_all_documents()
    print(f"Retrieved {len(all_docs)} docs from vector store.")
    
    # Build graph
    graph_rag.build_graph([doc.page_content for doc in all_docs], clear=True)
    stats = graph_rag.get_graph_stats()
    print(f"Graph Stats: {stats}")
    
    if stats["nodes"] > 1:
        print("SUCCESS: Graph generated with multiple nodes!")
    else:
        print("FAILURE: Graph still has 0 or 1 node.")

if __name__ == "__main__":
    ingest_and_verify()

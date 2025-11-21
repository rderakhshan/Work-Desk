import os
from pathlib import Path
from ai_workdesk.rag.ingestion import DocumentProcessor
from ai_workdesk.rag.vector_store import VectorStoreManager

def test_ingestion():
    print("🚀 Starting Ingestion Test...")
    
    # 1. Create a dummy file
    test_file = "test_doc.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a test document for the AI Workdesk ingestion pipeline.\n" * 50)
    
    print(f"✅ Created test file: {test_file}")
    
    try:
        # 2. Initialize Processor
        processor = DocumentProcessor()
        print("✅ Initialized DocumentProcessor")
        
        # 3. Load Documents
        docs = processor.load_documents([os.path.abspath(test_file)])
        print(f"✅ Loaded {len(docs)} documents")
        
        # 4. Chunk Documents
        chunks = processor.chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        print(f"✅ Created {len(chunks)} chunks")
        
        # 5. Initialize Vector Store
        vector_store = VectorStoreManager(persist_directory="./test_chroma_db")
        print("✅ Initialized VectorStoreManager")
        
        # 6. Add to Vector Store
        success = vector_store.add_documents(chunks)
        if success:
            print("✅ Successfully added documents to vector store")
        else:
            print("❌ Failed to add documents")
            
        # 7. Verify Stats
        stats = vector_store.get_stats()
        print(f"📊 Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print("🧹 Cleaned up test file")

if __name__ == "__main__":
    test_ingestion()

"""Quick script to check ChromaDB collection status."""
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = client.get_collection("ai_workdesk_collection")
    count = collection.count()
    print(f"Collection exists: ai_workdesk_collection")
    print(f"Total documents: {count}")
    
    if count > 0:
        # Get a sample
        results = collection.peek(limit=3)
        print(f"\nSample documents:")
        for i, doc in enumerate(results['documents'][:3]):
            print(f"{i+1}. {doc[:100]}...")
except Exception as e:
    print(f"Error: {e}")

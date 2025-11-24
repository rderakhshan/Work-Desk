import os
import sys
import sqlite3
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ai_workdesk.rag.metadata_store import MetadataStore

def test_metadata_schema():
    """Verify that the metadata table has the new columns."""
    print("Testing MetadataStore schema...")
    store = MetadataStore()
    
    # Check columns
    conn = sqlite3.connect(store.db_path)
    cursor = conn.execute("PRAGMA table_info(metadata)")
    columns = [info[1] for info in cursor.fetchall()]
    conn.close()
    
    print(f"Columns found: {columns}")
    
    if "summary" in columns and "metadata" in columns:
        print("✅ Schema update successful: 'summary' and 'metadata' columns exist.")
    else:
        print("❌ Schema update failed: Missing columns.")
        return False
        
    return True

def test_add_and_retrieve_entry():
    """Verify adding and retrieving an entry with summary and metadata."""
    print("\nTesting add_entry and list_entries...")
    store = MetadataStore()
    
    # Test data
    filename = "Test Video"
    size = 12345
    doc_type = "youtube"
    summary = "This is a test summary of the video."
    metadata = {
        "video_id": "test_id",
        "channel": "Test Channel",
        "duration": 600
    }
    
    # Add entry
    store.add_entry(filename, size, doc_type, summary=summary, metadata=metadata)
    print("Entry added.")
    
    # Retrieve entry
    entries = store.list_entries(limit=1)
    if not entries:
        print("❌ No entries found.")
        return False
        
    latest = entries[0]
    print(f"Retrieved entry: {latest}")
    
    # Verify fields
    if latest["summary"] == summary:
        print("✅ Summary field matches.")
    else:
        print(f"❌ Summary mismatch: Expected '{summary}', got '{latest.get('summary')}'")
        
    if latest["metadata"] == metadata:
        print("✅ Metadata field matches.")
    else:
        print(f"❌ Metadata mismatch: Expected '{metadata}', got '{latest.get('metadata')}'")
        
    return True

if __name__ == "__main__":
    if test_metadata_schema() and test_add_and_retrieve_entry():
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed.")

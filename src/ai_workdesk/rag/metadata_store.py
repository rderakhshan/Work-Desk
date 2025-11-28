import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
from loguru import logger


class MetadataStore:
    """SQLite-backed store for document ingestion metadata.

    Each entry records the filename, file size in bytes, upload timestamp (UTC ISO format),
    and a simple document type derived from the file extension.
    """

    def __init__(self, db_path: str = "data/metadata.db"):
        self.db_path = db_path
        logger.info(f"Initializing MetadataStore with database at: {os.path.abspath(self.db_path)}")
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the metadata table if it does not exist and ensure schema is up to date."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Create table with initial schema
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    upload_ts TEXT NOT NULL,
                    doc_type TEXT NOT NULL
                )
                """
            )
            
            # Check for new columns and add them if missing (migration)
            cursor = conn.execute("PRAGMA table_info(metadata)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if "summary" not in columns:
                conn.execute("ALTER TABLE metadata ADD COLUMN summary TEXT")
            
            if "metadata" not in columns:
                conn.execute("ALTER TABLE metadata ADD COLUMN metadata TEXT")
                
            conn.commit()
        finally:
            conn.close()

    def add_entry(self, filename: str, size: int, doc_type: str, summary: str = "", metadata: Dict = None) -> None:
        """Insert a new metadata record.

        Args:
            filename: Name of the uploaded file (basename).
            size: File size in bytes.
            doc_type: Simple document type (e.g., "txt", "pdf", "md").
            summary: Optional summary of the document.
            metadata: Optional additional metadata dictionary.
        """
        import json
        upload_ts = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO metadata (filename, size, upload_ts, doc_type, summary, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (filename, size, upload_ts, doc_type, summary, metadata_json),
            )
            conn.commit()
        finally:
            conn.close()

    def list_entries(self, offset: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        """Return a paginated list of metadata entries.

        Args:
            offset: Number of rows to skip.
            limit: Maximum number of rows to return.
        """
        import json
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT id, filename, size, upload_ts, doc_type, summary, metadata FROM metadata ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                meta_dict = {}
                try:
                    if row[6]:
                        meta_dict = json.loads(row[6])
                except json.JSONDecodeError:
                    pass
                    
                results.append({
                    "id": row[0],
                    "filename": row[1],
                    "size": row[2],
                    "upload_ts": row[3],
                    "doc_type": row[4],
                    "summary": row[5] or "",
                    "metadata": meta_dict
                })
            return results
        finally:
            conn.close()

    def delete_entry(self, entry_id: int) -> None:
        """Delete a metadata entry by its primary key."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM metadata WHERE id = ?", (entry_id,))
            conn.commit()
        finally:
            conn.close()

    def total_count(self) -> int:
        """Return total number of metadata records."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM metadata")
            (count,) = cursor.fetchone()
            return count
        finally:
            conn.close()

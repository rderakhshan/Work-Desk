import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any


class MetadataStore:
    """SQLite-backed store for document ingestion metadata.

    Each entry records the filename, file size in bytes, upload timestamp (UTC ISO format),
    and a simple document type derived from the file extension.
    """

    def __init__(self, db_path: str = "metadata.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the metadata table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        try:
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
            conn.commit()
        finally:
            conn.close()

    def add_entry(self, filename: str, size: int, doc_type: str) -> None:
        """Insert a new metadata record.

        Args:
            filename: Name of the uploaded file (basename).
            size: File size in bytes.
            doc_type: Simple document type (e.g., "txt", "pdf", "md").
        """
        upload_ts = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO metadata (filename, size, upload_ts, doc_type) VALUES (?, ?, ?, ?)",
                (filename, size, upload_ts, doc_type),
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
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT id, filename, size, upload_ts, doc_type FROM metadata ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "filename": row[1],
                    "size": row[2],
                    "upload_ts": row[3],
                    "doc_type": row[4],
                }
                for row in rows
            ]
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

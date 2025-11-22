import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ai_workdesk.rag.metadata_store import MetadataStore
from ai_workdesk.ui.gradio_app import AIWorkdeskUI

class TestMetadataStore(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_metadata.db"
        self.store = MetadataStore(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError:
                pass

    def test_crud(self):
        # Add
        self.store.add_entry("test.txt", 100, ".txt")
        self.assertEqual(self.store.total_count(), 1)
        
        # List
        entries = self.store.list_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["filename"], "test.txt")
        
        # Delete
        entry_id = entries[0]["id"]
        self.store.delete_entry(entry_id)
        self.assertEqual(self.store.total_count(), 0)

class TestUI(unittest.TestCase):
    @patch("ai_workdesk.ui.gradio_app.get_settings")
    @patch("ai_workdesk.ui.gradio_app.get_auth_manager")
    @patch("ai_workdesk.ui.gradio_app.MetadataStore")
    @patch("ai_workdesk.ui.gradio_app.DocumentProcessor")
    @patch("ai_workdesk.ui.gradio_app.VectorStoreManager")
    def test_ui_metadata(self, mock_vsm, mock_dp, mock_ms, mock_auth, mock_settings):
        # Setup mock store
        store_instance = mock_ms.return_value
        store_instance.list_entries.return_value = [
            {"id": 1, "filename": "file1.txt", "size": 1024, "upload_ts": "2023-01-01", "doc_type": ".txt"},
            {"id": 2, "filename": "file2.pdf", "size": 2048, "upload_ts": "2023-01-02", "doc_type": ".pdf"}
        ]
        store_instance.total_count.return_value = 25 # 2 pages if size 20
        
        # Mock settings
        mock_settings.return_value.openai_api_key = "fake-key"
        
        ui = AIWorkdeskUI()
        ui.page_size = 20
        
        # Test load_metadata
        data, max_page = ui.load_metadata(page=1)
        self.assertEqual(len(data), 2)
        self.assertEqual(max_page, 2)
        store_instance.list_entries.assert_called_with(limit=20, offset=0)
        
        # Test delete_metadata
        ui.delete_metadata(1, 1)
        store_instance.delete_entry.assert_called_with(1)
        # It calls load_metadata again
        self.assertEqual(store_instance.list_entries.call_count, 2)

if __name__ == "__main__":
    unittest.main()

"""
Launch AI Workdesk Web UI

Simple launcher script for the Gradio interface.
Run with: uv run python scripts/launch_ui.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_workdesk.ui.gradio_app import main

if __name__ == "__main__":
    main()

import sys
import os
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

def verify_imports():
    try:
        from ai_workdesk.tools.autogen_studio_manager import AutoGenStudioManager
        from ai_workdesk.ui.tabs.autogen import create_autogen_tab
        from ai_workdesk.ui.gradio_app import AIWorkdeskUI
        logger.info("✅ Imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def verify_manager():
    try:
        from ai_workdesk.tools.autogen_studio_manager import AutoGenStudioManager
        manager = AutoGenStudioManager(port=8082) # Use different port for test
        logger.info(f"✅ Manager instantiated. URL: {manager.get_ui_url()}")
        
        # We won't start the server here to avoid side effects/zombies in this simple test
        # but we can check if methods exist
        assert hasattr(manager, "start_server")
        assert hasattr(manager, "stop_server")
        return True
    except Exception as e:
        logger.error(f"❌ Manager verification failed: {e}")
        return False

if __name__ == "__main__":
    if verify_imports() and verify_manager():
        print("Verification passed!")
    else:
        print("Verification failed!")

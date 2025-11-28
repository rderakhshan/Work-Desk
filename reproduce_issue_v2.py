
import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath("src"))

# Mock heavy dependencies BEFORE importing gradio_app
sys.modules["ai_workdesk.rag.ingestion"] = MagicMock()
sys.modules["ai_workdesk.rag.vector_store"] = MagicMock()
sys.modules["ai_workdesk.rag.graph_rag"] = MagicMock()
sys.modules["ai_workdesk.rag.visualization"] = MagicMock()
sys.modules["ai_workdesk.rag.advanced_features"] = MagicMock()
sys.modules["ai_workdesk.tools.llm.ollama_client"] = MagicMock()
sys.modules["ai_workdesk.rag.metadata_store"] = MagicMock()
sys.modules["langchain_core.documents"] = MagicMock()
sys.modules["langchain_openai"] = MagicMock()

# Now import the class to test
from ai_workdesk.ui.gradio_app import AIWorkdeskUI
import plotly.graph_objects as go

def test_visualization():
    print("Initializing UI with mocks...")
    
    # Mock the visualizer instance on the class or instance
    # Since we can't easily patch __init__ without importing it, we'll instantiate and then patch
    
    # We need to mock get_settings and get_auth_manager too
    sys.modules["ai_workdesk.core.config"] = MagicMock()
    sys.modules["ai_workdesk.core.auth"] = MagicMock()
    
    # Re-import to apply mocks
    import ai_workdesk.ui.gradio_app
    from ai_workdesk.ui.gradio_app import AIWorkdeskUI
    
    ui = AIWorkdeskUI()
    
    # Setup the mock visualizer
    mock_fig = go.Figure()
    mock_fig.update_layout(title="Mock Figure")
    
    # Mock project_embeddings_2d to return (projected, fig)
    ui.visualizer.project_embeddings_2d.return_value = ([], mock_fig)
    
    print("Testing handle_visualization with PCA 2D...")
    try:
        fig = ui.handle_visualization("PCA", "2D")
        print(f"Return type: {type(fig)}")
        
        if isinstance(fig, tuple):
            print("ERROR: Returned a tuple!")
            print(f"Tuple content: {fig}")
        elif isinstance(fig, go.Figure):
            print("SUCCESS: Returned a go.Figure")
        else:
            print(f"WARNING: Returned unexpected type: {type(fig)}")
            
    except Exception as e:
        print(f"Exception during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization()


import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from ai_workdesk.ui.gradio_app import AIWorkdeskUI
import plotly.graph_objects as go

def test_visualization():
    print("Initializing UI...")
    ui = AIWorkdeskUI()
    
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

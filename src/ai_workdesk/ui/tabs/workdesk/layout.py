"""
Workdesk tab component for AI Workdesk UI.
Aggregates all sub-tabs for the Workdesk page.
"""
import gradio as gr
from ai_workdesk.ui.tabs.workdesk.ingestion import create_ingestion_tab
from ai_workdesk.ui.tabs.workdesk.rag_lab import create_rag_lab_tab
from ai_workdesk.ui.tabs.workdesk.chat_lab import create_chat_lab_tab
from ai_workdesk.ui.tabs.workdesk.visualization import create_visualization_tab
from ai_workdesk.ui.tabs.workdesk.metadata import create_metadata_tab

def create_workdesk_tab(ui):
    """
    Create the Workdesk tab content.
    
    Args:
        ui: The AIWorkdeskUI instance
        
    Returns:
        The workdesk page Group component
    """
    with gr.Group(visible=False) as workdesk_page:
        with gr.Tabs():
            # TAB 1: Embedding LAB
            with gr.TabItem("🧬 Embedding LAB"):
                with gr.Tabs():
                    create_ingestion_tab(ui)
                    create_metadata_tab(ui)
                    create_visualization_tab(ui)
            
            # TAB 2: RAG LAB
            create_rag_lab_tab(ui)
            
            # TAB 3: Chat LAB
            create_chat_lab_tab(ui)
            
    return workdesk_page

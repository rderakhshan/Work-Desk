"""
Home tab component for AI Workdesk UI.
"""
import gradio as gr
from ai_workdesk.smart_dashboard.ui import render_dashboard

def create_home_tab(demo):
    """
    Create the Home tab content.
    
    Args:
        demo: The Gradio Blocks instance (needed for loading events)
        
    Returns:
        The home page Group component
    """
    with gr.Group(visible=True) as home_page:
        dashboard_html, refresh_fn = render_dashboard()
        # Auto-load dashboard data
        demo.load(fn=refresh_fn, outputs=dashboard_html)
        
    return home_page

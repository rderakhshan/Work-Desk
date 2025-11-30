import gradio as gr
from ai_workdesk.tools.autogen_studio_manager import AutoGenStudioManager

def create_autogen_tab(manager: AutoGenStudioManager) -> gr.Blocks:
    """Creates the AutoGen Studio tab."""
    
    with gr.Column(elem_classes=["autogen-tab"]):
        gr.Markdown("## 🤖 AutoGen Studio")
        gr.Markdown("Create and manage multi-agent workflows using AutoGen Studio.")
        
        # Embed via iframe
        # We use a refresh button to reload the iframe if needed
        
        iframe_html = f"""
        <iframe 
            src="{manager.get_ui_url()}" 
            width="100%" 
            height="85vh" 
            style="border:none; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); min-height: 700px;"
        ></iframe>
        """
        
        html_component = gr.HTML(value=iframe_html)
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh AutoGen Studio", size="sm", variant="secondary")
            fullscreen_btn = gr.Button("↗️ Open Full Screen", size="sm", variant="primary")
            
        def refresh_view():
            return iframe_html
            
        refresh_btn.click(refresh_view, None, html_component)
        fullscreen_btn.click(None, None, None, js=f"""() => {{
            const url = '{manager.get_ui_url()}';
            const win = window.open('', '_blank');
            win.document.write(`
                <html>
                <head>
                    <title>AutoGen Studio - Full Screen</title>
                    <style>
                        body {{ margin: 0; padding: 0; overflow: hidden; }}
                        iframe {{ width: 100vw; height: 100vh; border: none; display: block; }}
                    </style>
                </head>
                <body>
                    <iframe src="${{url}}"></iframe>
                </body>
                </html>
            `);
        }}""")
        
    return html_component

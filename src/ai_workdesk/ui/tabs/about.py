"""
About tab component for AI Workdesk UI.
"""
import gradio as gr

def create_about_tab(settings):
    """
    Create the About tab content.
    
    Args:
        settings: The application settings object
        
    Returns:
        The about page Group component
    """
    with gr.Group(visible=False) as about_page:
        gr.Markdown("# ℹ️ About AI Workdesk")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div style="
                        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
                        border-radius: 15px;
                        padding: 20px;
                        border: 1px solid rgba(99, 102, 241, 0.3);
                        margin-bottom: 20px;
                    ">
                        <h3 style="color: #6366f1;">🚀 Modern Stack</h3>
                        <p style="color: #666;">Built with Gradio 5.0, LangChain, and UV</p>
                    </div>
                    """
                )
            
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border-radius: 15px;
                        padding: 20px;
                        border: 1px solid rgba(102, 126, 234, 0.3);
                        margin-bottom: 20px;
                    ">
                        <h3 style="color: #667eea;">🔐 Secure Auth</h3>
                        <p style="color: #666;">User login system for security</p>
                    </div>
                    """
                )

            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border-radius: 15px;
                        padding: 20px;
                        border: 1px solid rgba(102, 126, 234, 0.3);
                        margin-bottom: 20px;
                    ">
                        <h3 style="color: #667eea;">⚙️ Configurable</h3>
                        <p style="color: #666;">Adjust models and parameters</p>
                    </div>
                    """
                )

        gr.Markdown(
            """
            ## Version 0.1.0

            ### Supported Services
            - OpenAI (GPT-4, GPT-3.5)
            - Anthropic (Claude) - Coming Soon
            - Google AI - Coming Soon

            ### Tech Stack
            - **Framework**: Gradio 5.0
            - **Backend**: Python with UV package manager
            - **AI Services**: OpenAI, LangChain
            - **Authentication**: Secure user sessions

            ### Quick Tips
            1. Adjust temperature for creativity (0 = focused, 2 = creative)
            2. Use higher max tokens for longer responses
            3. Check Home page for API configuration status
            4. Clear chat history for a fresh start

            ---

            **Made with ❤️ using AI Workdesk**
            """
        )

        # Configuration Table
        gr.Markdown("## 🔧 Current Configuration")

        gr.DataFrame(
            value=[
                ["Default Model", settings.default_llm_model],
                ["Temperature", str(settings.default_temperature)],
                ["Max Tokens", str(settings.max_tokens)],
                ["Environment", settings.environment],
                ["Log Level", settings.log_level],
                ["Log File", str(settings.log_file)],
            ],
            headers=["Setting", "Value"],
            label="System Configuration",
        )
        
    return about_page

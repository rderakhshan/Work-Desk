"""
About tab component for AI Workdesk UI.
Comprehensive documentation with auto-generated TOC.
"""
import gradio as gr
from ai_workdesk.ui.utils.doc_renderer import DocumentationRenderer, COPY_CODE_JS


def create_about_tab(settings):
    """
    Create the About tab content with comprehensive documentation.
    
    Args:
        settings: The application settings object
        
    Returns:
        The about page Group component
    """
    # Initialize documentation renderer
    doc_renderer = DocumentationRenderer()
    
    with gr.Group(visible=False) as about_page:
        gr.Markdown("# 📚 AI Workdesk Documentation")
        
        # Documentation navigation tabs
        with gr.Tabs():
            # Tab 1: Overview
            with gr.TabItem("📖 Overview"):
                toc, content = doc_renderer.render("index.md")
                
                with gr.Row():
                    # Left: Table of Contents (20%)
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="color: #6366f1; margin-bottom: 15px;">📑 Contents</h3>
                            {toc}
                        </div>
                        """)
                    
                    # Right: Documentation Content (80%)
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 2: Getting Started
            with gr.TabItem("🚀 Getting Started"):
                toc, content = doc_renderer.render("getting-started.md")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="color: #6366f1; margin-bottom: 15px;">📑 Contents</h3>
                            {toc}
                        </div>
                        """)
                    
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 3: Features
            with gr.TabItem("✨ Features"):
                toc, content = doc_renderer.render("features.md")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="color: #6366f1; margin-bottom: 15px;">📑 Contents</h3>
                            {toc}
                        </div>
                        """)
                    
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 4: Configuration
            with gr.TabItem("⚙️ Configuration"):
                gr.Markdown("## 🔧 Current Configuration")
                
                gr.DataFrame(
                    value=[
                        ["Default Model", settings.default_llm_model],
                        ["Temperature", str(settings.default_temperature)],
                        ["Max Tokens", str(settings.max_tokens)],
                        ["Environment", settings.environment],
                        ["Log Level", settings.log_level],
                        ["Log File", str(settings.log_file)],
                        ["Embedding Provider", settings.default_embedding_provider],
                        ["Ollama Base URL", settings.ollama_base_url or "Not configured"],
                        ["Ollama Chat Model", settings.ollama_chat_model or "Not configured"],
                        ["Ollama Embedding Model", settings.ollama_embedding_model or "Not configured"],
                    ],
                    headers=["Setting", "Value"],
                    label="System Configuration",
                )
                
                gr.Markdown("""
                ### 📝 Configuration Files
                
                **Environment Variables:** `.env`  
                **Project Settings:** `pyproject.toml`  
                **Authentication:** `src/ai_workdesk/ui/gradio_app.py`
                
                ### 🔑 API Keys
                
                Configure in `.env`:
                ```bash
                # Cloud Models (optional)
                OPENAI_API_KEY=your-key-here
                
                # Local Models (recommended)
                OLLAMA_BASE_URL=http://localhost:11434
                OLLAMA_CHAT_MODEL=deepseek-r1:7b
                OLLAMA_EMBEDDING_MODEL=nomic-embed-text
                ```
                
                ### 🎯 Quick Settings
                
                - **Temperature:** Controls creativity (0 = focused, 2 = creative)
                - **Max Tokens:** Maximum response length
                - **Top-K:** Number of document chunks to retrieve
                - **Chunk Size:** Size of document chunks (512 recommended)
                - **Chunk Overlap:** Overlap between chunks (50 recommended)
                """)
            
            # Tab 5: About & Version
            with gr.TabItem("ℹ️ About"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
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
                        """)
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
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
                        """)

                    with gr.Column(scale=1):
                        gr.HTML("""
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
                        """)

                gr.Markdown("""
                ## Version 0.1.0

                ### Supported Services
                - ✅ Ollama (Local Models) - **Recommended**
                - ✅ OpenAI (GPT-4, GPT-3.5)
                - 🔜 Anthropic (Claude) - Coming Soon
                - 🔜 Google AI - Coming Soon

                ### Tech Stack
                - **Framework**: Gradio 5.0
                - **Backend**: Python 3.12+ with UV package manager
                - **AI Services**: Ollama, OpenAI, LangChain
                - **Vector Store**: ChromaDB
                - **Embeddings**: Ollama, OpenAI, HuggingFace
                - **Authentication**: Secure user sessions

                ### Quick Tips
                1. Use **Ollama** for privacy and offline usage
                2. Adjust **temperature** for creativity (0 = focused, 2 = creative)
                3. Use higher **max tokens** for longer responses
                4. Lower **similarity threshold** for more permissive retrieval (0.1-0.3)
                5. Check **Metadata** tab to see ingested documents

                ---

                **Made with ❤️ using AI Workdesk**  
                **GitHub:** [rderakhshan/Work-Desk](https://github.com/rderakhshan/Work-Desk)
                """)
        
        # Add copy code JavaScript
        gr.HTML(COPY_CODE_JS)
        
    return about_page

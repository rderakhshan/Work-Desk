"""
About tab component for AI Workdesk UI.
Professional documentation with auto-generated TOC.
"""
import gradio as gr
from ai_workdesk.ui.utils.doc_renderer import DocumentationRenderer, COPY_CODE_JS


def create_about_tab(settings):
    """
    Create the About tab content with professional documentation.
    
    Args:
        settings: The application settings object
        
    Returns:
        The about page Group component
    """
    # Initialize documentation renderer
    doc_renderer = DocumentationRenderer()
    
    with gr.Group(visible=False, elem_id="about-page-container") as about_page:
        # Professional Enterprise Header
        gr.HTML("""
        <div style="
            background: #ffffff;
            border-bottom: 1px solid #e1e4e8;
            padding: 32px 40px;
            margin-bottom: 24px;
        ">
            <!-- Breadcrumb -->
            <div style="
                font-size: 13px;
                color: #586069;
                margin-bottom: 16px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            ">
                <span style="color: #0366d6; text-decoration: none;">Documentation</span>
            </div>
            
            <!-- Main Title -->
            <h1 style="
                margin: 0 0 12px 0;
                font-size: 32px;
                font-weight: 600;
                color: #24292e;
                letter-spacing: -0.5px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            ">AI Workdesk Documentation</h1>
            
            <!-- Subtitle -->
            <p style="
                margin: 0;
                font-size: 16px;
                color: #586069;
                line-height: 1.5;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            ">Professional RAG platform for document processing and AI-powered retrieval</p>
        </div>
        """)
        
        # Documentation navigation tabs
        with gr.Tabs():
            # Tab 1: Overview
            with gr.TabItem("📖 Overview"):
                toc, content = doc_renderer.render("index.md")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="
                                color: #24292e;
                                font-size: 12px;
                                font-weight: 600;
                                text-transform: uppercase;
                                letter-spacing: 0.05em;
                                margin: 0 0 16px 0;
                                padding-bottom: 12px;
                                border-bottom: 1px solid #e5e7eb;
                            ">On This Page</h3>
                            <div style="font-size: 14px; line-height: 1.8;">
                                {toc}
                            </div>
                        </div>
                        """)
                    
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 2: Getting Started
            with gr.TabItem("🚀 Getting Started"):
                toc, content = doc_renderer.render("getting-started.md")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="
                                color: #24292e;
                                font-size: 12px;
                                font-weight: 600;
                                text-transform: uppercase;
                                letter-spacing: 0.05em;
                                margin: 0 0 16px 0;
                                padding-bottom: 12px;
                                border-bottom: 1px solid #e5e7eb;
                            ">On This Page</h3>
                            <div style="font-size: 14px; line-height: 1.8;">
                                {toc}
                            </div>
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
                            <h3 style="
                                color: #24292e;
                                font-size: 12px;
                                font-weight: 600;
                                text-transform: uppercase;
                                letter-spacing: 0.05em;
                                margin: 0 0 16px 0;
                                padding-bottom: 12px;
                                border-bottom: 1px solid #e5e7eb;
                            ">On This Page</h3>
                            <div style="font-size: 14px; line-height: 1.8;">
                                {toc}
                            </div>
                        </div>
                        """)
                    
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 4: Configuration
            with gr.TabItem("⚙️ Configuration"):
                toc, content = doc_renderer.render("configuration.md")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="
                                color: #24292e;
                                font-size: 12px;
                                font-weight: 600;
                                text-transform: uppercase;
                                letter-spacing: 0.05em;
                                margin: 0 0 16px 0;
                                padding-bottom: 12px;
                                border-bottom: 1px solid #e5e7eb;
                            ">On This Page</h3>
                            <div style="font-size: 14px; line-height: 1.8;">
                                {toc}
                            </div>
                        </div>
                        """)
                    
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 5: Advanced
            with gr.TabItem("🔬 Advanced"):
                toc, content = doc_renderer.render("advanced.md")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="
                                color: #24292e;
                                font-size: 12px;
                                font-weight: 600;
                                text-transform: uppercase;
                                letter-spacing: 0.05em;
                                margin: 0 0 16px 0;
                                padding-bottom: 12px;
                                border-bottom: 1px solid #e5e7eb;
                            ">On This Page</h3>
                            <div style="font-size: 14px; line-height: 1.8;">
                                {toc}
                            </div>
                        </div>
                        """)
                    
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 6: Troubleshooting
            with gr.TabItem("🔧 Troubleshooting"):
                toc, content = doc_renderer.render("troubleshooting.md")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="
                                color: #24292e;
                                font-size: 12px;
                                font-weight: 600;
                                text-transform: uppercase;
                                letter-spacing: 0.05em;
                                margin: 0 0 16px 0;
                                padding-bottom: 12px;
                                border-bottom: 1px solid #e5e7eb;
                            ">On This Page</h3>
                            <div style="font-size: 14px; line-height: 1.8;">
                                {toc}
                            </div>
                        </div>
                        """)
                    
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 7: API Reference
            with gr.TabItem("📚 API Reference"):
                toc, content = doc_renderer.render("api-reference.md")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["doc-sidebar"]):
                        gr.HTML(f"""
                        <div class="doc-toc-container">
                            <h3 style="
                                color: #24292e;
                                font-size: 12px;
                                font-weight: 600;
                                text-transform: uppercase;
                                letter-spacing: 0.05em;
                                margin: 0 0 16px 0;
                                padding-bottom: 12px;
                                border-bottom: 1px solid #e5e7eb;
                            ">On This Page</h3>
                            <div style="font-size: 14px; line-height: 1.8;">
                                {toc}
                            </div>
                        </div>
                        """)
                    
                    with gr.Column(scale=4, elem_classes=["doc-content"]):
                        gr.HTML(content)
            
            # Tab 8: System Information
            with gr.TabItem("ℹ️ System Info"):
                gr.HTML("""
                <div style="
                    background: #ffffff;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 24px;
                    margin-bottom: 24px;
                ">
                    <h2 style="
                        margin: 0 0 16px 0;
                        font-size: 20px;
                        font-weight: 600;
                        color: #111827;
                    ">Version Information</h2>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="padding: 12px 0; color: #6b7280; font-weight: 500;">Version</td>
                            <td style="padding: 12px 0; color: #111827;">0.1.0</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="padding: 12px 0; color: #6b7280; font-weight: 500;">Release Date</td>
                            <td style="padding: 12px 0; color: #111827;">November 2024</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="padding: 12px 0; color: #6b7280; font-weight: 500;">License</td>
                            <td style="padding: 12px 0; color: #111827;">MIT</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px 0; color: #6b7280; font-weight: 500;">Repository</td>
                            <td style="padding: 12px 0;">
                                <a href="https://github.com/rderakhshan/Work-Desk" 
                                   style="color: #2563eb; text-decoration: none;">
                                    github.com/rderakhshan/Work-Desk
                                </a>
                            </td>
                        </tr>
                    </table>
                </div>
                """)
                
                gr.Markdown("## Current Configuration")
                
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
                
                gr.HTML("""
                <div style="
                    background: #f9fafb;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 24px;
                ">
                    <h3 style="
                        margin: 0 0 12px 0;
                        font-size: 16px;
                        font-weight: 600;
                        color: #111827;
                    ">Supported Technologies</h3>
                    <ul style="
                        margin: 0;
                        padding-left: 20px;
                        color: #374151;
                        line-height: 1.8;
                    ">
                        <li><strong>Framework:</strong> Gradio 5.0</li>
                        <li><strong>Backend:</strong> Python 3.12+ with UV package manager</li>
                        <li><strong>LLM Providers:</strong> Ollama (local), OpenAI, Anthropic, Google AI</li>
                        <li><strong>Vector Store:</strong> ChromaDB with FAISS support</li>
                        <li><strong>Embeddings:</strong> Ollama, OpenAI, HuggingFace</li>
                        <li><strong>Authentication:</strong> Gradio built-in auth</li>
                    </ul>
                </div>
                """)
        
        # Add copy code JavaScript and enhanced styling
        gr.HTML(COPY_CODE_JS)
        
        # Add comprehensive modern styling
        gr.HTML("""
        <style>
            /* ============================================
               CREATIVE PROFESSIONAL DOCUMENTATION STYLES
               Scoped to #about-page-container
               ============================================ */
            
            /* Clean Page Background */
            #about-page-container {
                background: #ffffff;
                color: #24292e;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            }
            
            /* ============================================
               TOC STYLING
               ============================================ */
            
            #about-page-container .doc-toc-container {
                max-height: 65vh;
                overflow-y: auto;
                padding: 20px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                scrollbar-width: thin;
                scrollbar-color: #e1e4e8 transparent;
            }

            #about-page-container .doc-toc a {
                color: #586069;
                text-decoration: none;
                transition: all 0.2s ease;
                display: block;
                padding: 4px 0;
                font-size: 13px;
                border-left: 2px solid transparent;
                padding-left: 12px;
                margin-left: -14px;
            }
            
            #about-page-container .doc-toc a:hover {
                color: #0366d6;
                border-left-color: #0366d6;
            }
            
            #about-page-container .doc-toc ul {
                list-style: none;
                padding-left: 0;
                margin: 0;
            }
            
            #about-page-container .doc-toc li {
                margin: 4px 0;
            }
            
            #about-page-container .doc-toc ul ul {
                padding-left: 16px;
                margin-top: 4px;
            }
            
            /* Custom scrollbar for TOC */
            #about-page-container .doc-toc-container::-webkit-scrollbar {
                width: 6px;
            }
            
            #about-page-container .doc-toc-container::-webkit-scrollbar-track {
                background: transparent;
            }
            
            #about-page-container .doc-toc-container::-webkit-scrollbar-thumb {
                background: #e1e4e8;
                border-radius: 3px;
            }
            
            #about-page-container .doc-toc-container::-webkit-scrollbar-thumb:hover {
                background: #959da5;
            }
            
            /* ============================================
               CONTENT AREA ENHANCEMENTS
               ============================================ */
            
            /* Scrollable Content Container to keep tabs visible */
            #about-page-container .doc-content {
                padding: 0 24px 48px 24px;
                max-width: 900px;
                height: 65vh;
                overflow-y: auto;
                scrollbar-width: thin;
                scrollbar-color: #e1e4e8 transparent;
            }
            
            #about-page-container .doc-content::-webkit-scrollbar {
                width: 8px;
            }
            
            #about-page-container .doc-content::-webkit-scrollbar-track {
                background: transparent;
            }
            
            #about-page-container .doc-content::-webkit-scrollbar-thumb {
                background-color: #e1e4e8;
                border-radius: 4px;
            }
            
            /* Professional Headings */
            #about-page-container .doc-content h1 {
                color: #24292e;
                font-size: 32px;
                font-weight: 600;
                margin-top: 24px;
                margin-bottom: 24px;
                padding-bottom: 12px;
                border-bottom: 1px solid #eaecef;
                letter-spacing: -0.5px;
            }
            
            #about-page-container .doc-content h2 {
                color: #24292e;
                font-size: 24px;
                font-weight: 600;
                margin-top: 40px;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 1px solid #eaecef;
                letter-spacing: -0.3px;
            }
            
            #about-page-container .doc-content h3 {
                color: #24292e;
                font-size: 20px;
                font-weight: 600;
                margin-top: 32px;
                margin-bottom: 12px;
            }
            
            #about-page-container .doc-content p {
                color: #24292e;
                line-height: 1.6;
                margin-bottom: 16px;
                font-size: 16px;
            }
            
            /* Modern Code Styling */
            #about-page-container .doc-content code {
                background: rgba(27, 31, 35, 0.05);
                color: #24292e;
                padding: 0.2em 0.4em;
                border-radius: 6px;
                font-size: 85%;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            }
            
            #about-page-container .doc-content pre {
                background: #f6f8fa;
                padding: 16px;
                border-radius: 6px;
                overflow-x: auto;
                margin: 20px 0;
                border: 1px solid #e1e4e8;
            }
            
            #about-page-container .doc-content pre code {
                background: transparent;
                color: inherit;
                padding: 0;
                border: none;
                font-size: 100%;
            }
            
            /* Enhanced Lists */
            #about-page-container .doc-content ul, #about-page-container .doc-content ol {
                color: #24292e;
                line-height: 1.6;
                margin-bottom: 16px;
                padding-left: 24px;
            }
            
            #about-page-container .doc-content li {
                margin-bottom: 8px;
            }
            
            /* Links */
            #about-page-container .doc-content a {
                color: #0366d6;
                text-decoration: none;
                font-weight: 500;
            }
            
            #about-page-container .doc-content a:hover {
                text-decoration: underline;
            }
            
            /* Blockquotes */
            #about-page-container .doc-content blockquote {
                border-left: 4px solid #dfe2e5;
                color: #6a737d;
                padding: 0 16px;
                margin: 16px 0;
                background: transparent;
            }
            
            /* Tables */
            #about-page-container .doc-content table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                display: block;
                overflow-x: auto;
            }
            
            #about-page-container .doc-content th {
                font-weight: 600;
                padding: 12px 16px;
                border: 1px solid #dfe2e5;
                background-color: #f6f8fa;
            }
            
            #about-page-container .doc-content td {
                padding: 12px 16px;
                border: 1px solid #dfe2e5;
                background-color: #ffffff;
            }
            
            #about-page-container .doc-content tr:nth-child(2n) td {
                background-color: #f6f8fa;
            }
            
            /* Utility Classes */
            #about-page-container .doc-sidebar {
                border-right: 1px solid #e1e4e8;
                padding-right: 24px !important;
            }
            
            /* Sticky Tabs - Scoped to About Page */
            #about-page-container .tabs > .tab-nav {
                position: sticky;
                top: 0;
                z-index: 100;
                background: white;
                border-bottom: 1px solid #e1e4e8;
                margin-bottom: 0 !important;
            }

            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            #about-page-container .doc-content > * {
                animation: fadeIn 0.4s ease-out;
            }
        </style>
        """)
        
    return about_page

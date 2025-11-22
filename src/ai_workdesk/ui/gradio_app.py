"""
AI Workdesk - Gradio Web Interface with Authentication

A beautiful, secure web interface for your AI Workdesk with:
- User authentication
- Multiple AI service support
- Chat history
- Modern glassmorphism design
- Multi-page navigation with sidebar

Run with: uv run ai-workdesk-ui
"""

import os
from typing import List, Tuple
from ai_workdesk.rag.metadata_store import MetadataStore
import gradio as gr
from openai import OpenAI
from ai_workdesk.core.config import get_settings
from ai_workdesk.tools.llm.ollama_client import OllamaClient
from loguru import logger

from ai_workdesk import get_auth_manager, get_settings
from ai_workdesk.rag.ingestion import DocumentProcessor
from ai_workdesk.rag.vector_store import VectorStoreManager

# User credentials (in production, use a database)
USERS = {
    "admin": "admin123",  # Default user
    "demo": "demo123",
}


# Constants for selections
EMBEDDING_MODELS = ["OpenAI", "HuggingFace", "Ollama", "Google Gemini"]
DATABASES = ["ChromaDB", "FAISS", "PostgreSQL (PGVector)", "SQLite", "Pinecone"]

CUSTOM_CSS = """
/* Main Background - Pure White */
.gradio-container {
    background: #ffffff !important;
    min-height: 100vh !important; /* Force full viewport height */
    display: flex;
    flex-direction: column;
}

/* Force White on All Panels and Containers */
.glass-panel, .gray-panel, .panel {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
}

/* Tabs - Remove Default Grey */
.tabs, .tabitem {
    background: #ffffff !important;
    border: none !important;
}

/* Chatbot - White Background & Full Height */
.chat-container {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02) !important;
    height: 75vh !important; /* Dynamic viewport height */
    overflow-y: auto !important;
}

/* Primary Buttons - Indigo (Active State) */
.primary-btn {
    background: #6366f1 !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.3) !important;
}

/* Secondary Buttons - White (Inactive State) */
.secondary-btn {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #64748b !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.secondary-btn:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #334155 !important;
}

/* Inputs - White */
.gradio-dropdown, .gradio-slider, .gradio-textbox, .gradio-number {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Typography */
h1, h2, h3, h4 {
    color: #1e293b !important;
}
p, span, label {
    color: #475569 !important;
}

/* Status Box */
.status-box {
    background: #eff6ff !important;
    border: 1px solid #dbeafe !important;
    border-radius: 10px !important;
    padding: 12px !important;
    color: #1e40af !important;
}

/* Headers - Remove Grey */
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    background: transparent !important;
}

/* Tabs - White */
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
.tab-nav.selected {
    border-bottom: 2px solid #6366f1 !important;
    color: #6366f1 !important;
}

/* Sidebar Layout */
.sidebar-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 80vh; /* Match chat height roughly */
}
.sidebar-spacer {
    flex-grow: 1;
}
"""


class AIWorkdeskUI:
    """AI Workdesk Gradio UI with authentication and multi-page navigation."""

    def __init__(self):
        """Initialize the UI."""
        self.settings = get_settings()
        self.auth_manager = get_auth_manager()
        self.openai_client: OpenAI | None = None
        self._init_openai_client()
        # Initialize metadata store for document ingestion metadata
        self.metadata_store = MetadataStore()
        
        # Initialize RAG components
        self.doc_processor = DocumentProcessor()
        self._vector_store = None  # Lazy load to avoid blocking on model download
        self.page_size = 20  # Pagination size

    @property
    def vector_store(self):
        """Lazy load vector store with appropriate embedding provider based on settings."""
        if self._vector_store is None:
            logger.info("Initializing vector store (downloading model if needed)...")
            # Determine embedding provider: use Ollama if configured, else default to HuggingFace
            settings = get_settings()
            embedding_provider = "ollama" if getattr(settings, "ollama_embedding_model", None) else "huggingface"
            self._vector_store = VectorStoreManager(embedding_provider=embedding_provider)
        return self._vector_store

    def _init_openai_client(self):
        """Initialize OpenAI client with API key from settings."""
        if self.settings.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            logger.warning("OPENAI_API_KEY not found in .env file")
            self.openai_client = None

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate user.

        Args:
            username: Username
            password: Password

        Returns:
            True if authenticated, False otherwise
        """
        logger.info(f"Authentication attempt for user: {username}")

        if username in USERS and USERS[username] == password:
            logger.info(f"User {username} authenticated successfully")
            return True

        logger.warning(f"Failed authentication attempt for user: {username}")
        return False

    def get_auth_status(self) -> str:
        """Get formatted authentication status."""
        status = []
        
        # OpenAI Status
        if self.openai_client:
            status.append("✅ **OpenAI**: Connected")
        else:
            status.append("❌ **OpenAI**: Not Connected")
            
        return "\n".join(status)

    def load_metadata(self, page: int = 1) -> Tuple[List[List], int]:
        """Load metadata entries for the specified page."""
        # Ensure page is at least 1
        page = max(1, int(page))
        offset = (page - 1) * self.page_size
        entries = self.metadata_store.list_entries(limit=self.page_size, offset=offset)
        total_count = self.metadata_store.total_count()
        max_page = max(1, (total_count + self.page_size - 1) // self.page_size)
        
        # Format for Dataframe
        data = []
        for e in entries:
            # e is a dictionary
            data.append([e["id"], e["filename"], e["size"], e["upload_ts"], e["doc_type"]])
            
        return data, max_page

    def delete_metadata(self, entry_id: float, page: int) -> Tuple[List[List], int]:
        """Delete a metadata entry and refresh the list."""
        if entry_id is not None and entry_id > 0:
            self.metadata_store.delete_entry(int(entry_id))
        return self.load_metadata(page)

    def handle_ingestion(self, files, chunk_size, chunk_overlap) -> str:
        """Handle document ingestion."""
        if not files:
            return "⚠️ No files uploaded."
            
        try:
            file_paths = [f.name for f in files]
            
            # 1. Load Documents
            documents = self.doc_processor.load_documents(file_paths)
            if not documents:
                return "⚠️ No documents loaded."
                
            # 2. Record Metadata
            for f in files:
                try:
                    size = os.path.getsize(f.name)
                    filename = os.path.basename(f.name)
                    _, ext = os.path.splitext(filename)
                    self.metadata_store.add_entry(filename, size, ext)
                except Exception as e:
                    logger.error(f"Error recording metadata for {f.name}: {e}")

            # 3. Chunk Documents
            chunks = self.doc_processor.chunk_documents(
                documents, 
                chunk_size=int(chunk_size), 
                chunk_overlap=int(chunk_overlap)
            )
            
            # 4. Index in Vector Store
            self.vector_store.add_documents(chunks)
            
            return f"✅ Successfully ingested {len(files)} files ({len(chunks)} chunks)!"
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return f"❌ Error during ingestion: {str(e)}"

    def chat_with_ai(self, message, history, model, rag_technique, database, temperature, max_tokens, top_k, similarity_threshold, chunk_size, chunk_overlap, use_reranker, system_prompt):
        """Chat with AI using RAG."""
        if not message:
            return history, ""
        
        # Placeholder for actual RAG logic
        # In a real implementation, this would call self.vector_store.similarity_search
        # and then self.openai_client.chat.completions.create
        
        response = f"**Echo**: {message}\n\n*Model*: {model}\n*RAG*: {rag_technique}\n*Database*: {database}"
        
        # Append to history (Gradio 'messages' type uses list of dicts)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="AI Workdesk", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
            
            # State for navigation
            current_page = gr.State("home")
            
            with gr.Row():
                # Sidebar
                with gr.Column(scale=1, min_width=200, variant="panel", elem_classes=["glass-panel", "sidebar-container"]):
                    gr.Markdown("### 🧭 Navigation")
                    home_btn = gr.Button("🏠 Home", variant="primary", elem_classes=["primary-btn"])
                    workdesk_btn = gr.Button("🛠️ Work Desk", variant="secondary", elem_classes=["secondary-btn"])
                    about_btn = gr.Button("ℹ️ About", variant="secondary", elem_classes=["secondary-btn"])
                    
                    # Spacer to push logout to bottom
                    gr.HTML("<div style='flex-grow: 1; min-height: 200px;'></div>")
                    
                    gr.Markdown("---")
                    
                    # Logout Button (Bottom of sidebar)
                    with gr.Group():
                         logout_btn = gr.Button("🚪 Logout", variant="secondary", elem_classes=["secondary-btn"])
                    
                    gr.Markdown(self.get_auth_status())

                # Main Content
                with gr.Column(scale=5):
                    
                    # HOME PAGE
                    with gr.Group(visible=True) as home_page:
                        gr.Markdown("# 🏠 Welcome to AI Workdesk")
                        gr.Markdown("Your professional environment for AI engineering and RAG development.")
                        gr.Markdown("### Status")
                        gr.Markdown(self.get_auth_status())

                    # WORKDESK PAGE
                    with gr.Group(visible=False) as workdesk_page:
                        with gr.Tabs():
                            # TAB 1: Embedding LAB
                            with gr.TabItem("🧬 Embedding LAB"):
                                with gr.Tabs():
                                    with gr.TabItem("📤 Ingestion"):
                                        gr.Markdown("### 📄 Document Ingestion")
                                        file_input = gr.File(file_count="multiple", label="Upload Documents")
                                        with gr.Row():
                                            ingest_chunk_size = gr.Dropdown([256, 512, 1024], value=512, label="Chunk Size")
                                            ingest_chunk_overlap = gr.Slider(0, 200, 50, step=10, label="Overlap")
                                        ingest_btn = gr.Button("🚀 Ingest Documents", variant="primary", elem_classes=["primary-btn"])
                                        ingest_status = gr.Textbox(label="Status", interactive=False)
                                        
                                        ingest_btn.click(
                                            self.handle_ingestion,
                                            inputs=[file_input, ingest_chunk_size, ingest_chunk_overlap],
                                            outputs=[ingest_status]
                                        )

                                    with gr.TabItem("📋 Metadata"):
                                        gr.Markdown("### 🗄️ Ingested Files Metadata")
                                        with gr.Row():
                                            metadata_df = gr.Dataframe(
                                                headers=["ID", "Filename", "Size (bytes)", "Uploaded", "Type"],
                                                label="Ingested Files",
                                                interactive=False
                                            )
                                        with gr.Row():
                                            page_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="Page")
                                            refresh_btn = gr.Button("🔄 Refresh", elem_classes=["secondary-btn"])
                                        with gr.Row():
                                            delete_id = gr.Number(label="Entry ID to delete", precision=0)
                                            delete_btn = gr.Button("🗑️ Delete Entry", variant="stop", elem_classes=["secondary-btn"])
                                        
                                        # Callbacks
                                        refresh_btn.click(
                                            self.load_metadata,
                                            inputs=[page_slider],
                                            outputs=[metadata_df, page_slider]
                                        )
                                        page_slider.change(
                                            self.load_metadata,
                                            inputs=[page_slider],
                                            outputs=[metadata_df, page_slider]
                                        )
                                        delete_btn.click(
                                            self.delete_metadata,
                                            inputs=[delete_id, page_slider],
                                            outputs=[metadata_df, page_slider]
                                        )
                                        
                                        # Load initial data (on load)
                                        demo.load(self.load_metadata, inputs=[page_slider], outputs=[metadata_df, page_slider])

                            # TAB 2: Chat LAB
                            with gr.TabItem("💬 Chat LAB"):
                                with gr.Row(elem_classes=["chat-row"]):
                                    with gr.Column(scale=7):
                                        chatbot = gr.Chatbot(
                                            label="AI Assistant",
                                            show_copy_button=True,
                                            elem_classes=["chat-container"],
                                            type="messages",
                                            height=800,  # Increased height to fill screen
                                            render_markdown=True,
                                        )

                                        with gr.Row():
                                            msg = gr.Textbox(
                                                label="Message",
                                                placeholder="Ask me anything...",
                                                scale=4,
                                                show_label=False,
                                                container=False,
                                            )
                                            send_btn = gr.Button("Send 📤", scale=1, variant="primary", elem_classes=["primary-btn"])

                                        with gr.Row():
                                            clear_btn = gr.Button(
                                                "🗑️ Clear Chat", variant="secondary", scale=1, elem_classes=["secondary-btn"]
                                            )

                                    with gr.Column(scale=2, elem_classes=["glass-panel"]):
                                        model_dropdown = gr.Dropdown(
                                            choices=[
                                                "deepseek-r1:7b",
                                                "gemma3:4b",
                                                "llama3",
                                                "mistral",
                                                "phi3",
                                                "gpt-4o",
                                                "gpt-4o-mini",
                                                "gpt-4-turbo",
                                                "gpt-4",
                                                "gpt-3.5-turbo",
                                            ],
                                            value="deepseek-r1:7b",
                                            label="Model",
                                            allow_custom_value=True,
                                        )

                                        rag_dropdown = gr.Dropdown(
                                            choices=[
                                                "Naive RAG",
                                                "Hybrid Search",
                                                "Contextual RAG",
                                                "Graph RAG",
                                            ],
                                            value="Naive RAG",
                                            label="RAG Technique",
                                            allow_custom_value=True,
                                        )

                                        gr.Markdown(
                                            f"**Active Embedding:** {EMBEDDING_MODELS[2]}\\n\\n"
                                            "*Set during document ingestion in Embedding LAB*",
                                            elem_classes=["status-box"]
                                        )

                                        database_dropdown = gr.Dropdown(
                                            choices=DATABASES,
                                            value=DATABASES[0],
                                            label="Database",
                                            allow_custom_value=True,
                                        )

                                        temperature_slider = gr.Slider(
                                            minimum=0.0,
                                            maximum=2.0,
                                            value=self.settings.default_temperature,
                                            step=0.1,
                                            label="Temperature",
                                        )

                                        max_tokens_slider = gr.Slider(
                                            minimum=100,
                                            maximum=8192,
                                            value=min(self.settings.max_tokens, 8192),
                                            step=100,
                                            label="Max Tokens",
                                        )

                                        with gr.Accordion("⚙️ Advanced RAG Settings", open=False):
                                            gr.Markdown("### 🔍 Retrieval")
                                            top_k_slider = gr.Slider(
                                                minimum=1,
                                                maximum=20,
                                                value=5,
                                                step=1,
                                                label="Top-K (Chunks)",
                                            )
                                            similarity_threshold = gr.Slider(
                                                minimum=0.0,
                                                maximum=1.0,
                                                value=0.7,
                                                step=0.05,
                                                label="Similarity Threshold",
                                            )

                                            gr.Markdown("### 📄 Chunking")
                                            chunk_size = gr.Dropdown(
                                                choices=[256, 512, 1024, 2048],
                                                value=512,
                                                label="Chunk Size",
                                            )
                                            chunk_overlap = gr.Slider(
                                                minimum=0,
                                                maximum=200,
                                                value=50,
                                                step=10,
                                                label="Chunk Overlap",
                                            )

                                            gr.Markdown("### ⚡ Pipeline")
                                            use_reranker = gr.Checkbox(
                                                label="Use Reranker",
                                                value=False,
                                            )
                                            system_prompt = gr.Textbox(
                                                label="System Prompt",
                                                placeholder="You are a helpful AI assistant...",
                                                lines=3,
                                            )

                        # Chat interaction handlers
                        msg.submit(
                            self.chat_with_ai,
                            [
                                msg,
                                chatbot,
                                model_dropdown,
                                rag_dropdown,
                                database_dropdown,
                                temperature_slider,
                                max_tokens_slider,
                                top_k_slider,
                                similarity_threshold,
                                chunk_size,
                                chunk_overlap,
                                use_reranker,
                                system_prompt,
                            ],
                            [chatbot, msg],
                        )

                        send_btn.click(
                            self.chat_with_ai,
                            [
                                msg,
                                chatbot,
                                model_dropdown,
                                rag_dropdown,
                                database_dropdown,
                                temperature_slider,
                                max_tokens_slider,
                                top_k_slider,
                                similarity_threshold,
                                chunk_size,
                                chunk_overlap,
                                use_reranker,
                                system_prompt,
                            ],
                            [chatbot, msg],
                        )

                        clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

                    # About Page
                    with gr.Group(visible=False) as about_page:
                        gr.Markdown("# ℹ️ About AI Workdesk")

                        with gr.Row():
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
                                        <h3 style="color: #667eea;">💬 AI Chat</h3>
                                        <p style="color: #666;">Interactive conversations with GPT models</p>
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
                                ["Default Model", self.settings.default_llm_model],
                                ["Temperature", str(self.settings.default_temperature)],
                                ["Max Tokens", str(self.settings.max_tokens)],
                                ["Environment", self.settings.environment],
                                ["Log Level", self.settings.log_level],
                                ["Log File", str(self.settings.log_file)],
                            ],
                            headers=["Setting", "Value"],
                            label="System Configuration",
                        )

            # Navigation button handlers
            def show_home():
                return {
                    home_page: gr.update(visible=True),
                    workdesk_page: gr.update(visible=False),
                    about_page: gr.update(visible=False),
                    home_btn: gr.update(variant="primary", elem_classes=["primary-btn"]),
                    workdesk_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    about_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                }

            def show_workdesk():
                return {
                    home_page: gr.update(visible=False),
                    workdesk_page: gr.update(visible=True),
                    about_page: gr.update(visible=False),
                    home_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    workdesk_btn: gr.update(variant="primary", elem_classes=["primary-btn"]),
                    about_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                }

            def show_about():
                return {
                    home_page: gr.update(visible=False),
                    workdesk_page: gr.update(visible=False),
                    about_page: gr.update(visible=True),
                    home_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    workdesk_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    about_btn: gr.update(variant="primary", elem_classes=["primary-btn"]),
                }

            home_btn.click(
                show_home,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, about_btn],
            )

            workdesk_btn.click(
                show_workdesk,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, about_btn],
            )

            about_btn.click(
                show_about,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, about_btn],
            )
            
            # Logout handler (Javascript redirect)
            logout_btn.click(None, None, None, js="() => { window.location.href = '/logout'; }")
            
            # Footer
            gr.HTML(
                """
                <div style="
                    text-align: center;
                    padding: 30px 20px;
                    margin-top: 40px;
                    border-top: 1px solid rgba(226, 232, 240, 0.8);
                ">
                    <h2 style="
                        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-size: 2em;
                        font-weight: 800;
                        margin-bottom: 10px;
                        text-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
                    ">🚀 AI Work Desk</h2>
                    <p style="
                        color: #64748b;
                        font-size: 1.2em;
                        margin: 0;
                    ">Professional AI Engineering Environment</p>
                </div>
                """
            )

        return demo

    def launch(
        self,
        share: bool = False,
        server_port: int = 7860,
        auth: bool = True,
    ) -> None:
        """
        Launch the Gradio interface.

        Args:
            share: Create public link
            server_port: Port to run on
            auth: Enable authentication
        """
        logger.info(f"Launching AI Workdesk UI on port {server_port}")

        demo = self.create_interface()

        if auth:
            demo.launch(
                auth=self.authenticate,
                auth_message="🔐 Please login to AI Workdesk",
                share=share,
                server_port=server_port,
                favicon_path=None,
            )
        else:
            demo.launch(
                share=share,
                server_port=server_port,
                favicon_path=None,
            )


def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 60)
    print("🚀 AI Workdesk - Web Interface")
    print("=" * 60)
    print("\n📝 Default Login Credentials:")
    print("   Username: admin")
    print("   Password: admin123")
    print("\n   Username: demo")
    print("   Password: demo123")
    print("\n💡 Tip: Edit credentials in src/ai_workdesk/ui/gradio_app.py")
    print("=" * 60 + "\n")

    ui = AIWorkdeskUI()
    ui.launch(
        share=False,  # Set to True for public link
        server_port=7860,
        auth=True,  # Enable authentication
    )


# Expose demo for Gradio CLI auto-reload
if __name__ != "__main__":
    try:
        _ui = AIWorkdeskUI()
        demo = _ui.create_interface()
    except Exception:
        pass

if __name__ == "__main__":
    main()


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


class AIWorkdeskUI:
    """AI Workdesk Gradio UI with authentication and multi-page navigation."""

    def __init__(self):
        """Initialize the UI."""
        self.settings = get_settings()
        self.auth_manager = get_auth_manager()
        self.openai_client: OpenAI | None = None
        self._init_openai_client()
        
        # Initialize RAG components
        self.doc_processor = DocumentProcessor()
        self._vector_store = None  # Lazy load to avoid blocking on model download

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
        """Initialize OpenAI client with API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables")

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
            status.append("❌ **OpenAI**: Not Configured")
            
        return "\n\n".join(status)

    def handle_ingestion(self, files, chunk_size, chunk_overlap, embedding_model) -> dict:
        """
        Process uploaded files and ingest into vector store.
        
        Args:
            files: List of file objects from Gradio.
            chunk_size: Size of chunks.
            chunk_overlap: Overlap size.
            embedding_model: Selected embedding model.
            
        Returns:
            Updated status dictionary.
        """
        if not files:
            return {
                "status": "No files uploaded",
                "documents": 0,
                "chunks": 0,
                "vector_db": "ChromaDB"
            }
            
        try:
            logger.info(f"Starting ingestion: Model={embedding_model}, Chunk={chunk_size}, Overlap={chunk_overlap}")
            
            # Get file paths
            file_paths = [f.name for f in files]
            
            # 1. Load Documents
            docs = self.doc_processor.load_documents(file_paths)
            if not docs:
                return {"status": "Failed to load documents", "error": "No valid text found"}
                
            # 2. Chunk Documents
            chunks = self.doc_processor.chunk_documents(
                docs, 
                chunk_size=int(chunk_size), 
                chunk_overlap=int(chunk_overlap)
            )
            
            # 3. Add to Vector Store
            # Note: Currently using default model in VectorStoreManager. 
            # Future TODO: Pass embedding_model to VectorStoreManager
            success = self.vector_store.add_documents(chunks)
            
            if success:
                stats = self.vector_store.get_stats()
                return {
                    "status": "Success",
                    "documents": len(docs),
                    "new_chunks": len(chunks),
                    "total_chunks": stats.get("total_chunks", 0),
                    "vector_db": "ChromaDB"
                }
            else:
                return {"status": "Failed to store embeddings"}
                
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return {"status": "Error", "error": str(e)}

    def chat_with_ai(
        self,
        message: str,
        history: list,
        model: str,
        rag_technique: str,
        database_type: str,
        temperature: float,
        max_tokens: int,
        # Advanced Settings
        top_k: int,
        similarity_threshold: float,
        chunk_size: int,
        chunk_overlap: int,
        use_reranker: bool,
        system_prompt: str,
    ) -> tuple:
        """
        Chat with AI service.

        Args:
            message: User message
            history: Chat history
            model: Model name
            rag_technique: RAG technique to use
            database_type: Database type to use
            temperature: Temperature setting
            max_tokens: Maximum tokens
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks
            use_reranker: Whether to use reranking
            system_prompt: Custom system prompt

        Returns:
            Tuple of (updated chat history, empty string for message box)
        """
        if not message.strip():
            return history, ""
        
        try:
            # Add user message to history
            history.append({"role": "user", "content": message})
            
            # Step 1: Retrieve relevant documents from vector store
            logger.info(f"Retrieving documents for query: {message[:50]}...")
            relevant_docs = self.vector_store.similarity_search(
                query=message,
                k=top_k,
                score_threshold=similarity_threshold
            )
            
            # Step 2: Build context from retrieved documents
            if relevant_docs:
                context = "\n\n".join([
                    f"Document {i+1}:\n{doc.page_content}" 
                    for i, doc in enumerate(relevant_docs)
                ])
                logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
            else:
                context = "No relevant documents found in the knowledge base."
                logger.warning("No documents retrieved from vector store")
            
            # Step 3: Determine which model to use
            ollama_model = self.settings.ollama_chat_model
            if model and not model.startswith("gpt"):
                ollama_model = model
            
            # Step 4: Build RAG prompt
            rag_prompt = ""
            if system_prompt:
                rag_prompt += f"System Instructions: {system_prompt}\n\n"
            
            # Check if this is the first message (empty history except current user message)
            is_first_message = len(history) <= 1
            
            rag_prompt += f"""You are a helpful AI assistant. Answer the user's question based on the following context from the knowledge base.

Context from Knowledge Base:
{context}

User Question: {message}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Cite specific parts of the context when relevant
{"- IMPORTANT: Start your response by introducing yourself with: 'Hello! I am " + ollama_model + ", ready to help you with your questions based on the knowledge base.'" if is_first_message else ""}

Answer:"""
            
            # Step 5: Initialize Ollama client with parameters
            client = OllamaClient(
                model=ollama_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Step 5: Get RAG response
            logger.info(f"Generating RAG response with model: {ollama_model}")
            response = client.chat(rag_prompt)
            
            # Add assistant response to history
            history.append({"role": "assistant", "content": response})
            
            return history, ""
            
        except Exception as e:
            logger.error(f"RAG chat error: {e}")
            error_msg = f"Error: {str(e)}\n\nNote: Make sure you have ingested documents in the Embedding LAB first."
            history.append({"role": "assistant", "content": error_msg})
            return history, ""

    def create_interface(self) -> "gr.Blocks":
        custom_css = """
        .gradio-container {
            font-family: 'Inter', sans-serif;
            max-width: 95% !important;
        }

        /* Sidebar Styling */
        .sidebar-nav {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }

        .nav-button {
            width: 100%;
            margin: 5px 0;
            transition: all 0.3s ease;
        }

        .nav-button:hover {
            transform: translateX(5px);
        }

        /* Page Content Styling */
        .header-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: 800;
            text-align: center;
            margin: 20px 0;
        }

        .status-box {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }

        .chat-container {
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            height: 70vh !important;
            min-height: 500px;
        }

        /* Hide Gradio footer */
        footer {
            display: none !important;
        }

        /* Smooth transitions for page changes */
        .transition-fade {
            transition: opacity 0.3s ease-in-out;
        }
        """

        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="purple",
                neutral_hue="slate",
            ),
            css=custom_css,
            title="AI Workdesk",
        ) as demo:

            with gr.Row():
                # Sidebar
                with gr.Sidebar(open=True, width=250):
                    gr.Markdown("### 🧭 Navigation")

                    home_btn = gr.Button(
                        "🏠 Home",
                        variant="primary",
                        elem_classes=["nav-button"],
                        size="lg",
                    )

                    workdesk_btn = gr.Button(
                        "💼 Work Desk",
                        variant="secondary",
                        elem_classes=["nav-button"],
                        size="lg",
                    )

                    about_btn = gr.Button(
                        "ℹ️ About",
                        variant="secondary",
                        elem_classes=["nav-button"],
                        size="lg",
                    )

                    # Push subsequent content to bottom
                    with gr.Column(scale=1, min_width=0):
                        pass

                    # Logout Button
                    logout_btn = gr.Button(
                        "🚪 Logout",
                        variant="secondary",
                        elem_classes=["nav-button"],
                        size="lg",
                    )

                    logout_btn.click(
                        fn=None,
                        inputs=None,
                        outputs=None,
                        js="function() { window.location.href = '/logout'; }",
                    )

                    gr.Markdown("---")
                    gr.Markdown(
                        """
                        <div style="text-align: center; padding: 10px; color: #666; font-size: 0.9em;">
                            <p>v0.1.0</p>
                            <p>Powered by UV & Gradio</p>
                        </div>
                        """
                    )

                # Main Content Area
                with gr.Column(scale=4):
                    # Home Page
                    with gr.Column(visible=True) as home_page:
                        # Welcome Section
                        gr.HTML(
                            """
                            <div style="text-align: center; padding: 40px 20px;">
                                <h1 style="
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    -webkit-background-clip: text;
                                    -webkit-text-fill-color: transparent;
                                    font-size: 3.5em;
                                    font-weight: 800;
                                    margin-bottom: 10px;
                                ">🚀 Welcome to AI Workdesk</h1>
                                <p style="color: #666; font-size: 1.3em; margin-top: 10px;">
                                    Your Professional AI Development Environment
                                </p>
                            </div>
                            """
                        )

                        # Status Cards
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        border-radius: 15px;
                                        padding: 25px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        backdrop-filter: blur(10px);
                                        text-align: center;
                                    ">
                                        <div style="font-size: 3em; margin-bottom: 10px;">💬</div>
                                        <h3 style="color: #667eea; margin: 10px 0;">AI Chat</h3>
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
                                        padding: 25px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        backdrop-filter: blur(10px);
                                        text-align: center;
                                    ">
                                        <div style="font-size: 3em; margin-bottom: 10px;">🔐</div>
                                        <h3 style="color: #667eea; margin: 10px 0;">Secure</h3>
                                        <p style="color: #666;">Authentication and API key management</p>
                                    </div>
                                    """
                                )

                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        border-radius: 15px;
                                        padding: 25px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        backdrop-filter: blur(10px);
                                        text-align: center;
                                    ">
                                        <div style="font-size: 3em; margin-bottom: 10px;">⚙️</div>
                                        <h3 style="color: #667eea; margin: 10px 0;">Configurable</h3>
                                        <p style="color: #666;">Customize models and parameters</p>
                                    </div>
                                    """
                                )

                        gr.Markdown("---")

                        # API Status Section
                        gr.Markdown("## 📊 System Status")
                        auth_status = gr.Markdown(
                            self.get_auth_status(),
                            elem_classes=["status-box"],
                        )

                        with gr.Row():
                            refresh_status_btn = gr.Button(
                                "🔄 Refresh Status", variant="secondary", scale=1
                            )

                        refresh_status_btn.click(
                            lambda: self.get_auth_status(),
                            None,
                            auth_status,
                        )

                        gr.Markdown("---")

                        # Quick Actions
                        gr.Markdown("## 🚀 Quick Actions")
                        gr.Markdown(
                            "Ready to start working with AI? Head to the Work Desk to begin chatting!"
                        )

                    # Work Desk Page
                    with gr.Column(visible=False) as workdesk_page:
                        gr.Markdown("# 🧪 AI Engineering Labs")

                        with gr.Tabs():
                            # Tab 1: Embedding LAB
                            with gr.TabItem("🧬 Embedding LAB"):
                                gr.Markdown("### 📚 Knowledge Base Management")
                                gr.Markdown(
                                    "Upload documents, manage your vector database, and visualize embeddings here."
                                )
                                
                                with gr.Row():
                                    # Left Column - File Upload
                                    with gr.Column(scale=1):
                                        file_upload = gr.File(
                                            label="📄 Upload Documents",
                                            file_count="multiple",
                                            file_types=[".txt", ".pdf", ".md"],
                                            height=200,
                                        )
                                        ingest_btn = gr.Button(
                                            "📥 Ingest & Embed", 
                                            variant="primary",
                                            size="lg",
                                            scale=1
                                        )
                                    
                                    # Middle Column - Ingestion Settings
                                    with gr.Column(scale=1):
                                        gr.Markdown("#### ⚙️ Ingestion Settings")
                                        ingest_chunk_size = gr.Dropdown(
                                            choices=[256, 512, 1024, 2048],
                                            value=512,
                                            label="Chunk Size",
                                        )
                                        ingest_chunk_overlap = gr.Slider(
                                            minimum=0,
                                            maximum=200,
                                            value=50,
                                            step=10,
                                            label="Chunk Overlap",
                                        )
                                        ingest_embedding_model = gr.Dropdown(
                                            choices=EMBEDDING_MODELS,
                                            value=EMBEDDING_MODELS[2], # Default to Ollama
                                            label="Embedding Model",
                                        )
                                    
                                    # Right Column - Database Status
                                    with gr.Column(scale=1):
                                        gr.Markdown("#### 📊 Database Status")
                                        db_status = gr.JSON(
                                            value={
                                                "status": "Ready",
                                                "documents": 0,
                                                "chunks": 0,
                                                "vector_db": "ChromaDB"
                                            },
                                            label="Current State"
                                        )
                                        
                                # Connect Ingestion
                                ingest_btn.click(
                                    self.handle_ingestion,
                                    inputs=[
                                        file_upload, 
                                        ingest_chunk_size, 
                                        ingest_chunk_overlap,
                                        ingest_embedding_model
                                    ],
                                    outputs=[db_status]
                                )

                            # Tab 2: Chat LAB
                            with gr.TabItem("💬 Chat LAB"):
                                with gr.Row():
                                    with gr.Column(scale=7):  # Increased from 4 to 7
                                        chatbot = gr.Chatbot(
                                            label="AI Assistant",
                                            show_copy_button=True,
                                            elem_classes=["chat-container"],
                                            type="messages",
                                        )

                                        with gr.Row():
                                            msg = gr.Textbox(
                                                label="Message",
                                                placeholder="Ask me anything...",
                                                scale=4,
                                                show_label=False,
                                            )
                                            send_btn = gr.Button("Send 📤", scale=1, variant="primary")

                                        with gr.Row():
                                            clear_btn = gr.Button(
                                                "🗑️ Clear Chat", variant="secondary", scale=1
                                            )

                                    with gr.Column(scale=2):  # Reduced from 1 to 2 for better proportion
                                        model_dropdown = gr.Dropdown(
                                            choices=[
                                                # Ollama Models (Local)
                                                "deepseek-r1:7b",
                                                "gemma3:4b",
                                                "llama3",
                                                "mistral",
                                                "phi3",
                                                # OpenAI Models (Cloud)
                                                "gpt-4o",
                                                "gpt-4o-mini",
                                                "gpt-4-turbo",
                                                "gpt-4",
                                                "gpt-3.5-turbo",
                                            ],
                                            value="deepseek-r1:7b",  # Default to DeepSeek
                                            label="Model",
                                            allow_custom_value=True,  # Allow custom model names
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

                                        # Show active embedding model (read-only)
                                        gr.Markdown(
                                            f"**Active Embedding:** {EMBEDDING_MODELS[2]}\n\n"
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
                                            maximum=8192,  # Increased to support larger token limits
                                            value=min(
                                                self.settings.max_tokens, 8192
                                            ),  # Ensure value is within range
                                            step=100,
                                            label="Max Tokens",
                                        )

                                        # Advanced Settings Accordion
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
                    with gr.Column(visible=False) as about_page:
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
                    home_btn: gr.update(variant="primary"),
                    workdesk_btn: gr.update(variant="secondary"),
                    about_btn: gr.update(variant="secondary"),
                }

            def show_workdesk():
                return {
                    home_page: gr.update(visible=False),
                    workdesk_page: gr.update(visible=True),
                    about_page: gr.update(visible=False),
                    home_btn: gr.update(variant="secondary"),
                    workdesk_btn: gr.update(variant="primary"),
                    about_btn: gr.update(variant="secondary"),
                }

            def show_about():
                return {
                    home_page: gr.update(visible=False),
                    workdesk_page: gr.update(visible=False),
                    about_page: gr.update(visible=True),
                    home_btn: gr.update(variant="secondary"),
                    workdesk_btn: gr.update(variant="secondary"),
                    about_btn: gr.update(variant="primary"),
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
            
            # Footer
            gr.HTML(
                """
                <div style="
                    text-align: center;
                    padding: 30px 20px;
                    margin-top: 40px;
                    border-top: 2px solid rgba(102, 126, 234, 0.2);
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                ">
                    <h2 style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-size: 2em;
                        font-weight: 800;
                        margin-bottom: 10px;
                    ">🚀 AI Work Desk</h2>
                    <p style="
                        color: #666;
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


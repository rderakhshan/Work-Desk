print("!!!!!!!! GRADIO APP MODULE IS BEING RELOADED !!!!!!!!")
"""
AI Workdesk - Main Gradio Application
"""
import os
import gradio as gr
from loguru import logger
from typing import Optional, Tuple, List, Dict, Any

from ai_workdesk.core.config import get_settings
from ai_workdesk.core.auth import get_auth_manager
from ai_workdesk.rag.ingestion import DocumentProcessor
from ai_workdesk.rag.vector_store import VectorStoreManager
from ai_workdesk.rag.graph_rag import GraphRAG
from ai_workdesk.rag.visualization import EmbeddingVisualizer, analyze_cluster_quality, estimate_tokens
from ai_workdesk.rag.advanced_features import DataCleaner
from ai_workdesk.tools.llm.ollama_client import OllamaClient
from ai_workdesk.rag.metadata_store import MetadataStore
from ai_workdesk.rag.youtube_summarizer import YouTubeSummarizer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# UI Components
from ai_workdesk.ui.components.layout import create_sidebar_content, CUSTOM_CSS
from ai_workdesk.ui.tabs.home import create_home_tab
from ai_workdesk.ui.tabs.workdesk import create_workdesk_tab
from ai_workdesk.ui.tabs.about import create_about_tab
from ai_workdesk.tools.autogen_studio_manager import AutoGenStudioManager
from ai_workdesk.ui.constants import MODELS, EMBEDDING_MODELS, DATABASES
import atexit

class AIWorkdeskUI:
    """AI Workdesk Gradio UI with authentication and multi-page navigation."""

    def __init__(self):
        """Initialize the UI."""
        self.settings = get_settings()
        self.auth_manager = get_auth_manager()
        self.openai_client = None
        self._init_openai_client()
        
        # Initialize metadata store
        self.metadata_store = MetadataStore()
        
        # Initialize RAG components
        self.doc_processor = DocumentProcessor()
        
        # Initialize Visualizer
        logger.info("Initializing visualizer...")
        self.visualizer = EmbeddingVisualizer()
        
        # Initialize GraphRAG
        logger.info("Initializing GraphRAG...")
        self.graph_rag = GraphRAG()
        
        # Initialize Advanced Features
        logger.info("Initializing advanced features...")
        try:
            self.data_cleaner = DataCleaner()
        except Exception as e:
            logger.warning(f"DataCleaner initialization failed: {e}")
            self.data_cleaner = None
        
        # Initialize YouTube Summarizer
        self.youtube_summarizer = YouTubeSummarizer()

        # Initialize AutoGen Studio Manager
        self.autogen_manager = AutoGenStudioManager()
        try:
            self.autogen_manager.start_server()
            atexit.register(self.autogen_manager.stop_server)
        except Exception as e:
            logger.error(f"Failed to start AutoGen Studio: {e}")
        
        # Initialize Obsidian Exporter
        try:
            from ai_workdesk.tools.obsidian_exporter import ObsidianExporter
            vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
            
            # Fallback: Try known path if env var is missing
            if not vault_path:
                known_path = r"C:\Users\Riemann\Documents\AI Engineering\AI Engineering\AI Engeeneering"
                if os.path.exists(known_path):
                    vault_path = known_path
                    logger.info(f"Using auto-detected Obsidian vault: {vault_path}")
            
            if vault_path and os.path.exists(vault_path):
                self.obsidian_exporter = ObsidianExporter(vault_path)
                logger.info(f"Obsidian exporter initialized with vault: {vault_path}")
            else:
                self.obsidian_exporter = None
                logger.warning("Obsidian vault path not configured or doesn't exist")
        except Exception as e:
            logger.warning(f"Failed to initialize Obsidian exporter: {e}")
            self.obsidian_exporter = None

        self._vector_store = None  # Lazy load
        self.page_size = 20
        self._last_doc_hash = None  # Track document changes for graph caching

    def _init_openai_client(self):
        """Initialize OpenAI client if API key is available."""
        if self.settings.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            # Explicitly initialize with the configured default provider
            logger.info(f"Initializing VectorStoreManager with default provider: {self.settings.default_embedding_provider}")
            self._vector_store = VectorStoreManager(
                embedding_provider=self.settings.default_embedding_provider
            )
        return self._vector_store

    def authenticate(self, username, password):
        """Authenticate user."""
        return self.auth_manager.authenticate(username, password)

    def get_auth_status(self):
        """Get authentication status markdown."""
        # Simple status for now
        return "✅ Authenticated"

    def load_metadata(self, page=1):
        """Load metadata for display."""
        try:
            entries = self.metadata_store.list_entries(limit=self.page_size, offset=(page-1)*self.page_size)
            # Convert to list of lists for Dataframe
            data = []
            for entry in entries:
                data.append([
                    entry['id'],
                    entry['filename'],
                    entry['doc_type'],
                    f"{entry['size']/1024:.1f} KB",
                    entry['upload_ts'],
                    0 # Chunk count placeholder
                ])
            return data, f"Page {page}"
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return [], f"Error: {e}"

    def delete_metadata(self, entry_id, page=1):
        """Delete metadata entry."""
        try:
            if not entry_id:
                return self.load_metadata(page)
            self.metadata_store.delete_entry(entry_id)
            return self.load_metadata(page)
        except Exception as e:
            logger.error(f"Error deleting metadata: {e}")
            return [], f"Error: {e}"

    def create_collection(self, name):
        if not name:
            return "⚠️ Please enter a collection name", gr.Dropdown()
        success = self.vector_store.create_collection(name)
        if success:
            collections = self.vector_store.list_collections()
            return f"✅ Created collection: {name}", gr.Dropdown(choices=collections)
        return f"❌ Failed to create collection: {name}", gr.Dropdown()

    def switch_collection(self, name):
        if not name:
            return "⚠️ Please select a collection"
        success = self.vector_store.switch_collection(name)
        if success:
            self._invalidate_graph_cache()  # Invalidate cache when switching collections
            return f"✅ Switched to collection: {name}"
        return f"❌ Failed to switch to collection: {name}"

    def delete_collection(self, name):
        if not name:
            return "⚠️ Please select a collection", gr.Dropdown()
        if name == self.vector_store.collection_name:
            return "⚠️ Cannot delete active collection", gr.Dropdown()
        success = self.vector_store.delete_collection(name)
        if success:
            collections = self.vector_store.list_collections()
            return f"✅ Deleted collection: {name}", gr.Dropdown(choices=collections)
        return f"❌ Failed to delete collection: {name}", gr.Dropdown()

    def update_models(self, provider):
        models = MODELS.get(provider, MODELS["Ollama"])
        return gr.update(choices=models, value=models[0])

    def chat_router(self, chat_input, history, model, rag_technique, provider, temperature, max_tokens, top_k, similarity_threshold, chunk_size, chunk_overlap, use_reranker, system_prompt, enable_voice_response):
        """Route to appropriate chat handler."""
        # Extract text and files from MultimodalTextbox input
        if isinstance(chat_input, dict):
            message = chat_input.get("text", "")
            attached_files = chat_input.get("files", [])
        else:
            # Fallback for string input (e.g. from voice or tests)
            message = str(chat_input)
            attached_files = []

        if attached_files:
            return self.chat_with_attached_document(
                message, history, attached_files, model, temperature, max_tokens, chunk_size, chunk_overlap, system_prompt
            )
        else:
            return self.chat_with_ai(
                message, history, model, rag_technique, "ChromaDB", temperature, max_tokens, top_k, similarity_threshold, chunk_size, chunk_overlap, use_reranker, system_prompt, enable_voice_response
            )

    def chat_with_attached_document(self, message, history, files, model, temperature, max_tokens, chunk_size, chunk_overlap, system_prompt):
        """Chat with attached documents."""
        # Placeholder for actual implementation
        # For now, just return a simple response
        updated_history = history + [[message, f"I received your files: {[os.path.basename(f) for f in files]}. (Document chat not fully implemented yet)"]]
        return None, updated_history  # Clear input, update chat

    def handle_ingestion(self, files, chunk_size, chunk_overlap, strategy):
        """Handle document ingestion."""
        if not files:
            return "⚠️ Please upload files."
        
        try:
            # Process documents
            documents = self.doc_processor.process_files(
                [f.name for f in files],
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap)
            )
            
            if not documents:
                return "❌ No text could be extracted from the files."
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            self._invalidate_graph_cache()  # Invalidate graph cache when documents change
            
            
            # Save metadata
            for f in files:
                try:
                    self.metadata_store.add_entry(
                        filename=os.path.basename(f.name),
                        size=os.path.getsize(f.name),
                        doc_type=os.path.splitext(f.name)[1][1:]
                    )
                except Exception as e:
                    logger.warning(f"Failed to save metadata for {f.name}: {e}")
            
            return f"✅ Successfully ingested {len(files)} files ({len(documents)} chunks)."
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return f"❌ Error: {str(e)}"

    def handle_web_ingestion(self, url, depth, chunk_size, chunk_overlap):
        """Handle web ingestion."""
        if not url:
            return "⚠️ Please enter a URL."
        
        try:
            # Process web content
            documents = self.doc_processor.process_web(
                url,
                depth=int(depth),
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap)
            )
            
            if not documents:
                return "❌ No content could be extracted."
            
            self.vector_store.add_documents(documents)
            
            # Save metadata
            try:
                total_size = sum(len(doc.page_content) for doc in documents)
                self.metadata_store.add_entry(
                    filename=url,
                    size=total_size,
                    doc_type="web"
                )
            except Exception as e:
                logger.warning(f"Failed to save metadata for {url}: {e}")
            
            return f"✅ Successfully crawled {url} ({len(documents)} chunks)."
        except Exception as e:
            logger.error(f"Web ingestion error: {e}")
            return f"❌ Error: {str(e)}"

    def handle_youtube_ingestion(self, urls, chunk_size, chunk_overlap, generate_summary, support_playlists):
        """Handle YouTube ingestion."""
        if not urls:
            return "⚠️ Please enter YouTube URLs."
        
        try:
            url_list = [u.strip() for u in urls.split('\n') if u.strip()]
            
            documents = self.doc_processor.load_youtube_documents(url_list)
            
            if not documents:
                return "❌ No transcripts found."
            
            self.vector_store.add_documents(documents)
            self._invalidate_graph_cache()  # Invalidate graph cache when documents change
            
            # Save metadata for each URL
            try:
                for url in url_list:
                    # Find the documents that came from this URL
                    url_docs = [d for d in documents if d.metadata.get('url') == url]
                    if url_docs:
                        total_size = sum(len(doc.page_content) for doc in url_docs)
                        # Use the title from the first document's metadata as the filename
                        filename = url_docs[0].metadata.get('source', url)
                        self.metadata_store.add_entry(
                            filename=filename,
                            size=total_size,
                            doc_type="youtube"
                        )
            except Exception as e:
                logger.warning(f"Failed to save metadata for YouTube URLs: {e}")

            return f"✅ Successfully processed {len(url_list)} videos ({len(documents)} chunks)."
        except Exception as e:
            logger.error(f"YouTube ingestion error: {e}")
            return f"❌ Error: {str(e)}"

    def handle_youtube_summarization(self, url):
        """Handle YouTube video summarization."""
        if not url:
            return "⚠️ Please enter a YouTube URL.", None, None
        
        try:
            logger.info(f"Starting summarization for {url}")
            # Get video content
            content = self.youtube_summarizer.get_video_content(url)
            text = content['text']
            logger.info(f"Retrieved transcript length: {len(text)} chars")
            
            # Generate summary
            summary = self.youtube_summarizer.generate_summary(text)
            logger.info(f"Generated summary length: {len(summary)} chars")
            
            return summary, text, [] # Return summary, full text (hidden state), and empty chat history
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return f"❌ Error: {str(e)}", None, None

    def handle_youtube_chat(self, message, history, video_text):
        """Handle chat with YouTube video."""
        if not message:
            return "", history
            
        if not video_text:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "⚠️ Please summarize a video first to load its content."})
            return "", history
            
        try:
            response = self.youtube_summarizer.chat_with_video(video_text, message, history)
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return "", history
        except Exception as e:
            logger.error(f"YouTube chat error: {e}")
            history.append({{"role": "user", "content": message}})
            history.append({{"role": "assistant", "content": f"❌ Error: {str(e)}"}})
            return "", history

    def handle_obsidian_export(self, url, summary, video_text, chat_history):
        """Export YouTube video summary and chat to Obsidian vault."""
        if not self.obsidian_exporter:
            return "❌ Obsidian exporter not configured. Please set OBSIDIAN_VAULT_PATH in your .env file."
        
        if not url:
            return "⚠️ Please enter a YouTube URL first."
        
        if not summary:
            return "⚠️ Please summarize the video first before exporting."
        
        try:
            # Extract video ID and get metadata
            from ai_workdesk.rag.youtube_loader import YouTubeTranscriptLoader
            loader = YouTubeTranscriptLoader()
            video_id = loader._parse_video_id(url)
            
            if not video_id:
                return "❌ Invalid YouTube URL."
            
            # Get video metadata
            metadata = loader.fetch_metadata(video_id)
            
            # Format chat history if available
            chat_text = None
            if chat_history and len(chat_history) > 0:
                chat_lines = []
                for msg in chat_history:
                    role = msg.get("role", "").upper()
                    content = msg.get("content", "")
                    if role == "USER":
                        chat_lines.append(f"**Q:** {content}")
                    elif role == "ASSISTANT":
                        chat_lines.append(f"**A:** {content}\n")
                chat_text = "\n".join(chat_lines)
            
            # Export to Obsidian
            note_path = self.obsidian_exporter.export_youtube_note(
                video_title=metadata.get('source', 'YouTube Video'),
                video_url=url,
                channel=metadata.get('channel', 'Unknown'),
                summary=summary,
                transcript=video_text if video_text else None,
                chat_history=chat_text,
                metadata={
                    'duration': metadata.get('duration', 0),
                    'view_count': metadata.get('view_count', 0)
                }
            )
            
            return f"✅ Successfully exported to Obsidian!\n📝 Note saved at: {note_path}"
            
        except Exception as e:
            logger.error(f"Obsidian export error: {e}")
            return f"❌ Export failed: {str(e)}"

    def handle_chat_obsidian_export(self, chat_history):
        """Export generic chat history to Obsidian vault."""
        if not self.obsidian_exporter:
            return "❌ Obsidian exporter not configured. Please set OBSIDIAN_VAULT_PATH in your .env file."
        
        if not chat_history:
            return "⚠️ Chat history is empty."
        
        try:
            # Format chat history
            chat_lines = []
            first_user_msg = "Generic Chat"
            
            for msg in chat_history:
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
                
                # Use first user message as title (truncated)
                if role == "USER" and first_user_msg == "Generic Chat":
                    first_user_msg = content[:50] + "..." if len(content) > 50 else content
                
                if role == "USER":
                    chat_lines.append(f"**Q:** {content}")
                elif role == "ASSISTANT":
                    chat_lines.append(f"**A:** {content}\n")
            
            chat_text = "\n".join(chat_lines)
            
            # Export to Obsidian
            note_path = self.obsidian_exporter.export_chat_note(
                title=first_user_msg,
                chat_history=chat_text
            )
            
            return f"✅ Successfully exported to Obsidian!\n📝 Note saved at: {note_path}"
            
        except Exception as e:
            logger.error(f"Obsidian export error: {e}")
            return f"❌ Export failed: {str(e)}"

    def handle_rag_obsidian_export(self, chat_history):
        """Export RAG chat history to Obsidian vault."""
        if not self.obsidian_exporter:
            return "❌ Obsidian exporter not configured. Please set OBSIDIAN_VAULT_PATH in your .env file."
        
        if not chat_history:
            return "⚠️ Chat history is empty."
        
        try:
            # Format chat history
            # For RAG, we want to capture the last Q&A pair as the main content, 
            # but also save the full history.
            
            chat_lines = []
            last_query = "RAG Session"
            last_answer = ""
            
            for msg in chat_history:
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
                
                if role == "USER":
                    chat_lines.append(f"**Q:** {content}")
                    last_query = content
                elif role == "ASSISTANT":
                    chat_lines.append(f"**A:** {content}\n")
                    last_answer = content
            
            chat_text = "\n".join(chat_lines)
            
            # Truncate title
            title = last_query[:50] + "..." if len(last_query) > 50 else last_query
            
            # Export to Obsidian
            note_path = self.obsidian_exporter.export_rag_note(
                title=title,
                query=last_query,
                answer=last_answer,
                chat_history=chat_text
            )
            
            return f"✅ Successfully exported to Obsidian!\n📝 Note saved at: {note_path}"
            
        except Exception as e:
            logger.error(f"Obsidian export error: {e}")
            return f"❌ Export failed: {str(e)}"

    def handle_audio_transcription(self, audio_file, language):
        """Handle audio transcription."""
        if not audio_file:
            return "", "⚠️ No audio file."
        # Simplified placeholder
        return "Transcription placeholder", "✅ Transcribed (Placeholder)"

    def handle_audio_ingestion(self, transcription, chunk_size, chunk_overlap):
        """Handle audio ingestion."""
        if not transcription:
            return "⚠️ No transcription."
        # Simplified placeholder
        return "✅ Ingested (Placeholder)"

    def handle_voice_query(self, audio_file):
        """
        Handle voice query transcription.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with text and empty files list for MultimodalTextbox
        """
        if not audio_file:
            return {"text": "", "files": []}
        
        try:
            from faster_whisper import WhisperModel
            
            # Use small model for speed
            model = WhisperModel("small", device="cpu", compute_type="int8")
            
            # Transcribe
            segments, info = model.transcribe(audio_file, beam_size=5)
            
            # Combine all segments
            transcription = " ".join([segment.text for segment in segments])
            
            logger.info(f"Transcribed audio: {transcription[:100]}...")
            
            # Return in MultimodalTextbox format
            return {"text": transcription, "files": []}
            
        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            return {"text": f"❌ Transcription failed: {str(e)}", "files": []}

    def handle_voice_response(self, text):
        """Handle voice response."""
        return None

    def handle_visualization(self, method, dimension):
        """Handle visualization.

        Generates an embedding visualization using the selected reduction method and dimension.
        Returns a Plotly Figure (compatible with gr.Plot).
        """
        try:
            # Lazy import to avoid heavy dependencies at module load
            import numpy as np
            import plotly.graph_objects as go

            # Fetch real embeddings from vector store
            data = self.vector_store.get_all_embeddings()
            embeddings = data.get("embeddings")
            metadatas = data.get("metadatas")
            
            if embeddings is None or len(embeddings) == 0:
                 # Fallback to dummy if empty
                 logger.warning("No embeddings found, using dummy data")
                 embeddings = np.random.rand(100, 768)
                 labels = [f"Doc {i}" for i in range(100)]
            else:
                 embeddings = np.array(embeddings)
                 # Create labels from metadata (e.g., source filename)
                 labels = []
                 if metadatas:
                     for i, m in enumerate(metadatas):
                         source = m.get("source", "")
                         if source:
                             # Use basename of source
                             import os
                             label = os.path.basename(source)
                         else:
                             label = f"Doc {i}"
                         labels.append(label)
                 else:
                     labels = [f"Doc {i}" for i in range(len(embeddings))]

            # Use the fetched/dummy embeddings
            dummy_embeddings = embeddings
            dummy_labels = labels

            # Choose projection based on dimension
            if dimension == "2D":
                projected, fig = self.visualizer.project_embeddings_2d(
                    dummy_embeddings, labels=dummy_labels, method=method.lower()
                )
            else:  # 3D
                projected, fig = self.visualizer.project_embeddings_3d(
                    dummy_embeddings, labels=dummy_labels, method=method.lower()
                )

            # Ensure we have a Plotly Figure
            if isinstance(fig, tuple):
                # If visualizer returns (projected, fig), extract fig
                if len(fig) >= 2:
                    fig = fig[1]
                else:
                    # Fallback if tuple structure is unexpected
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.update_layout(title="Visualization Error: Unexpected return format")

            if not isinstance(fig, go.Figure):
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.update_layout(title="Visualization Failed: Invalid return type")

            return fig
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            import plotly.graph_objects as go
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Error: {str(e)}")
            return empty_fig

    def handle_token_estimation(self, files):
        """Handle token estimation."""
        return "💰 Cost estimation (Placeholder)"

    def handle_graph_generation(self, max_nodes=100, min_weight=1, viz_mode="2D", bg_color="#ffffff"):
        """Handle graph generation."""
        try:
            # Fetch all documents from the vector store
            all_docs = self.vector_store.get_all_documents()

            if not all_docs:
                return "<div><p style='text-align:center; padding:20px;'>No documents in the vector store. Please ingest some documents first.</p></div>"

            # Calculate hash of current documents to detect changes
            current_doc_hash = hash(tuple(sorted([doc.page_content[:100] for doc in all_docs])))
            
            # Only rebuild graph if documents changed
            if self._last_doc_hash != current_doc_hash:
                logger.info(f"Documents changed, rebuilding graph...")
                self.graph_rag.build_graph([doc.page_content for doc in all_docs], clear=True)
                self._last_doc_hash = current_doc_hash
            else:
                logger.info("Using cached graph, only updating visualization parameters...")
            
            # Check graph stats first
            stats = self.graph_rag.get_graph_stats()
            if stats["nodes"] < 2:
                return f"<div style='text-align:center; padding:20px;'><p>Graph is too small to display. It requires at least 2 entities to form a connection.</p><p><b>Current Nodes: {stats['nodes']}</b></p><p>Please try ingesting more documents with diverse entities.</p></div>"

            # Generate graph HTML with filters and mode
            html_path = self.graph_rag.visualize_graph(max_nodes=int(max_nodes), min_edge_weight=int(min_weight), mode=viz_mode)
            
            if not html_path or not os.path.exists(html_path):
                return "<div><p style='text-align:center; padding:20px;'>No graph data available. Please ingest some documents first.</p></div>"
            
            # Read HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Wrap in iframe for isolation to ensure scripts run correctly
            import html
            escaped_html = html.escape(html_content)
            # Use user-selected background color
            iframe_html = f'<iframe srcdoc="{escaped_html}" width="100%" height="1000px" style="border:none; background-color:{bg_color};"></iframe>'
                
            return iframe_html
        except Exception as e:
            logger.error(f"Graph generation error: {e}")
            return f"<div><p style='color:red;'>Error generating graph: {str(e)}</p></div>"

    def get_graph_html_path(self, max_nodes=100, min_weight=1, viz_mode="2D"):
        """Generate graph and return file path for full screen view."""
        try:
            # Handle None values from Gradio (use defaults)
            max_nodes = max_nodes if max_nodes is not None else 100
            min_weight = min_weight if min_weight is not None else 1
            viz_mode = viz_mode if viz_mode is not None else "2D"
            
            # Ensure graph is built (reuse existing logic if possible, but for now just ensure it exists)
            all_docs = self.vector_store.get_all_documents()
            if not all_docs:
                return None
                
            current_doc_hash = hash(tuple(sorted([doc.page_content[:100] for doc in all_docs])))
            
            if self._last_doc_hash != current_doc_hash:
                self.graph_rag.build_graph([doc.page_content for doc in all_docs], clear=True)
                self._last_doc_hash = current_doc_hash
            
            # Generate graph HTML
            html_path = self.graph_rag.visualize_graph(max_nodes=int(max_nodes), min_edge_weight=int(min_weight), mode=viz_mode)
            return html_path
        except Exception as e:
            logger.error(f"Error getting graph path: {e}")
            return None

    def chat_with_ai(self, message, history, model, rag_technique, database, temperature, max_tokens, top_k, similarity_threshold, chunk_size, chunk_overlap, use_reranker, system_prompt, enable_voice_response=False):
        """Chat with AI using Ollama or OpenAI."""
        try:
            # Build conversation history
            messages = []
            
            # Add system prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history (limit to last 10 messages to avoid context overflow)
            for msg in history[-10:]:
                if isinstance(msg, dict):
                    messages.append(msg)
                elif isinstance(msg, list) and len(msg) == 2:
                    # Old format: [user_msg, assistant_msg]
                    messages.append({"role": "user", "content": msg[0]})
                    messages.append({"role": "assistant", "content": msg[1]})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Get response from LLM
            if model.startswith("gpt-"):
                # OpenAI Models
                if not hasattr(self, 'openai_client') or not self.openai_client:
                    self._init_openai_client()
                
                if not hasattr(self, 'openai_client') or not self.openai_client:
                     raise ValueError("OpenAI client not initialized. Please check your API key in settings or .env file.")

                response_obj = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response = response_obj.choices[0].message.content
            else:
                # Ollama Models
                from ai_workdesk.tools.llm.ollama_client import OllamaClient
                
                ollama_client = OllamaClient(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                response = ollama_client.chat(messages, model=model)
            
            # Update history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return None, history  # Clear input, update chat
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = f"❌ Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return None, history

    def export_chat(self, history):
        """Export chat."""
        # The JS handles the download, this just needs to return something or nothing
        return None

    def clear_vector_store(self):
        """Clear the current vector store collection."""
        try:
            self.vector_store.delete_collection(self.vector_store.collection_name)
            self._invalidate_graph_cache()  # Invalidate cache when clearing store
            return f"✅ Successfully cleared collection: {self.vector_store.collection_name}"
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return f"❌ Error clearing vector store: {str(e)}"

    def _invalidate_graph_cache(self):
        """Invalidate graph cache when documents change."""
        self._last_doc_hash = None
        logger.info("Graph cache invalidated")

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="AI Workdesk", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
            
            # State for navigation
            current_page = gr.State("home")
            
            with gr.Row():
                # Sidebar
                with gr.Column(scale=1, min_width=200, variant="panel", elem_classes=["glass-panel", "sidebar-container"]):
                    home_btn, workdesk_btn, autogen_btn, about_btn, logout_btn = create_sidebar_content()
                    gr.Markdown(self.get_auth_status())
                    
                    # Logo at very bottom
                    import os
                    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
                    gr.Image(
                        value=logo_path,
                        show_label=False,
                        show_download_button=False,
                        container=False,
                        elem_classes=["sidebar-logo"]
                    )

                # Main Content
                with gr.Column(scale=5, elem_classes=["main-content"]):
                    
                    # HOME PAGE
                    home_page = create_home_tab(demo)

                    # WORKDESK PAGE
                    workdesk_page = create_workdesk_tab(self)

                    # ABOUT PAGE
                    about_page = create_about_tab(self.settings)

            # Navigation button handlers
            def show_home():
                return {
                    home_page: gr.update(visible=True),
                    workdesk_page: gr.update(visible=False),
                    about_page: gr.update(visible=False),
                    home_btn: gr.update(variant="primary", elem_classes=["primary-btn"]),
                    workdesk_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    about_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    autogen_btn: gr.update(variant="secondary", elem_classes=["secondary-btn", "external-link-btn"]),
                }

            def show_workdesk():
                return {
                    home_page: gr.update(visible=False),
                    workdesk_page: gr.update(visible=True),
                    about_page: gr.update(visible=False),
                    home_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    workdesk_btn: gr.update(variant="primary", elem_classes=["primary-btn"]),
                    about_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    autogen_btn: gr.update(variant="secondary", elem_classes=["secondary-btn", "external-link-btn"]),
                }

            def show_about():
                return {
                    home_page: gr.update(visible=False),
                    workdesk_page: gr.update(visible=False),
                    about_page: gr.update(visible=True),
                    home_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    workdesk_btn: gr.update(variant="secondary", elem_classes=["secondary-btn"]),
                    about_btn: gr.update(variant="primary", elem_classes=["primary-btn"]),
                    autogen_btn: gr.update(variant="secondary", elem_classes=["secondary-btn", "external-link-btn"]),
                }

            home_btn.click(
                show_home,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, autogen_btn, about_btn],
            )

            workdesk_btn.click(
                show_workdesk,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, autogen_btn, about_btn],
            )

            about_btn.click(
                show_about,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, autogen_btn, about_btn],
            )
            
            # AutoGen Studio handler (opens in new tab)
            autogen_btn.click(
                None,
                None,
                None,
                js="() => { window.open('http://localhost:8081', '_blank'); }"
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


# Create UI instance and demo globally for Gradio hot reloading
ui = AIWorkdeskUI()
demo = ui.create_interface()

# Explicitly set auth details on the Blocks object for Gradio CLI
demo.auth = ui.authenticate
demo.auth_message = "🔐 Please login to AI Workdesk"

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

    # Launch using the globally created demo
    demo.launch(
        auth=ui.authenticate,
        auth_message="🔐 Please login to AI Workdesk",
        share=False,
        server_port=7860,
        favicon_path=None,
    )

if __name__ == "__main__":
    main()

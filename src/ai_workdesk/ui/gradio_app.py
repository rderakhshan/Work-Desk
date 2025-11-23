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
import gradio as gr
from openai import OpenAI
from loguru import logger

from ai_workdesk.core.config import get_settings
from ai_workdesk import get_auth_manager
from ai_workdesk.tools.llm.ollama_client import OllamaClient
from ai_workdesk.rag.metadata_store import MetadataStore
from ai_workdesk.rag.ingestion import DocumentProcessor
from ai_workdesk.rag.vector_store import VectorStoreManager
from ai_workdesk.rag.visualization import EmbeddingVisualizer, estimate_tokens, analyze_cluster_quality
from ai_workdesk.rag.graph_rag import GraphRAG
from ai_workdesk.smart_dashboard.ui import render_dashboard, DASHBOARD_CSS

# User credentials (in production, use a database)
USERS = {
    "admin": "admin123",  # Default user
    "demo": "demo123",
}


# Constants for selections
EMBEDDING_MODELS = ["OpenAI", "HuggingFace", "Ollama", "Google Gemini"]
DATABASES = ["ChromaDB", "FAISS", "PostgreSQL (PGVector)", "SQLite", "Pinecone"]

# Provider-based model lists
MODELS = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    "Ollama": ["deepseek-r1:7b", "gemma3:4b", "llama3", "mistral", "phi3"]
}

CUSTOM_CSS = """
/* Project Ambitions: Mesh Gradient & Global Reset */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

:root, .dark, body, gradio-app {
    --font-sans: 'Outfit', sans-serif !important;
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --block-background-fill: transparent !important;
    --border-color-primary: transparent !important;
    --body-background-fill: transparent !important;
}

/* Animated Mesh Gradient Background - REMOVED per user request */
body {
    background-color: #ffffff !important;
    margin: 0;
    padding: 0;
    overflow: hidden;
    font-family: 'Outfit', sans-serif !important;
}

.gradio-container {
    background: #ffffff !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    display: flex;
    flex-direction: column;
}

/* Sidebar - Glass & Borderless */
.sidebar-container {
    height: 100vh !important;
    overflow-y: auto !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
}

/* Main Content - Full Width & Transparent */
.main-content {
    height: 100vh !important;
    overflow-y: auto !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Force Transparency on Panels */
.glass-panel, .gray-panel, .panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Tabs - Clean */
.tabs, .tabitem {
    background: transparent !important;
    border: none !important;
}

/* Chatbot - Floating Glass */
.chat-container {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    height: 75vh !important;
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

/* Button Active States */
.primary-btn:active, .secondary-btn:active {
    background: #6366f1 !important;
    color: white !important;
    transform: scale(0.98);
}

/* Title Fix - Bring to Front with Maximum Specificity */
.gradio-container h1,
.gradio-container h2,
h1, h2 {
    position: relative !important;
    z-index: 1000 !important;
    background: white !important;
    padding: 10px 0 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Logo Styling - 80% Sidebar Width */
.sidebar-logo {
    text-align: center;
    margin-top: 20px;
    padding: 15px;
    width: 100%;
}
.sidebar-logo img {
    width: 80% !important;
    height: auto !important;
    max-width: 180px;
    border-radius: 20px;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    transition: transform 0.2s ease;
}
.sidebar-logo img:hover {
    transform: scale(1.05);
}
""" + DASHBOARD_CSS


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
        
        # Initialize Visualizer
        logger.info("Initializing visualizer...")
        self.visualizer = EmbeddingVisualizer()
        
        # Initialize GraphRAG
        logger.info("Initializing GraphRAG...")
        self.graph_rag = GraphRAG()
        
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
        """Load metadata entries for the specified page, showing only unique filenames."""
        # Ensure page is at least 1
        page = max(1, int(page))
        
        # Get all entries and deduplicate by filename (keep most recent)
        all_entries = self.metadata_store.list_entries(limit=1000, offset=0)
        
        # Group by filename, keep only the most recent entry for each
        unique_entries = {}
        for e in all_entries:
            filename = e["filename"]
            if filename not in unique_entries or e["upload_ts"] > unique_entries[filename]["upload_ts"]:
                unique_entries[filename] = e
        
        # Convert to list and sort by upload time (most recent first)
        entries_list = sorted(unique_entries.values(), key=lambda x: x["upload_ts"], reverse=True)
        
        # Paginate
        total_count = len(entries_list)
        max_page = max(1, (total_count + self.page_size - 1) // self.page_size)
        offset = (page - 1) * self.page_size
        paginated_entries = entries_list[offset:offset + self.page_size]
        
        # Format for Dataframe
        data = []
        for e in paginated_entries:
            data.append([e["id"], e["filename"], e["size"], e["upload_ts"], e["doc_type"]])
            
        return data, max_page

    def delete_metadata(self, entry_id: float, page: int) -> Tuple[List[List], int]:
        """Delete a metadata entry and refresh the list."""
        if entry_id is not None and entry_id > 0:
            self.metadata_store.delete_entry(int(entry_id))
        return self.load_metadata(page)

    def handle_ingestion(self, files, chunk_size, chunk_overlap, chunking_strategy="Fixed Size") -> str:
        """Handle document ingestion with configurable chunking strategy."""
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

            # 3. Chunk Documents (Fixed or Semantic)
            if chunking_strategy == "Semantic":
                chunks = self.doc_processor.chunk_documents_semantic(documents)
            else:  # Fixed Size
                chunks = self.doc_processor.chunk_documents(
                    documents, 
                    chunk_size=int(chunk_size), 
                    chunk_overlap=int(chunk_overlap)
                )
            
            # 4. Index in Vector Store
            self.vector_store.add_documents(chunks)
            
            strategy_msg = "semantic chunking" if chunking_strategy == "Semantic" else f"{chunk_size}-token chunks"
            return f"✅ Successfully ingested {len(files)} files ({len(chunks)} chunks using {strategy_msg})!"
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return f"❌ Error during ingestion: {str(e)}"

    def handle_web_ingestion(self, url, depth, chunk_size, chunk_overlap) -> str:
        """Handle web document ingestion."""
        if not url:
            return "⚠️ Please enter a URL."
            
        try:
            # 1. Load Documents
            documents = self.doc_processor.load_web_documents(url, int(depth))
            if not documents:
                return f"⚠️ No documents found at {url}."
                
            # 2. Record Metadata (for the main URL)
            try:
                self.metadata_store.add_entry(url, 0, ".html")
            except Exception as e:
                logger.error(f"Error recording metadata for {url}: {e}")

            # 3. Chunk Documents
            chunks = self.doc_processor.chunk_documents(
                documents, 
                chunk_size=int(chunk_size), 
                chunk_overlap=int(chunk_overlap)
            )
            
            # 4. Index in Vector Store
            self.vector_store.add_documents(chunks)
            
            return f"✅ Successfully ingested {len(documents)} pages from {url} ({len(chunks)} chunks)!"
        except Exception as e:
            logger.error(f"Web ingestion error: {e}")
            return f"❌ Error during web ingestion: {str(e)}"

    def handle_visualization(self, method, dimension) -> Tuple[str, str, str]:
        """Handle embedding visualization."""
        try:
            # 1. Get all embeddings
            data = self.vector_store.get_all_embeddings()
            if not data or not data.get("embeddings"):
                return "⚠️ No embeddings found in collection.", "", ""
            
            embeddings = data["embeddings"]
            metadatas = data["metadatas"]
            
            # Prepare labels
            labels = []
            for m in metadatas:
                labels.append(f"{m.get('filename', 'Unknown')} (Chunk {m.get('chunk_index', '?')})")
            
            # 2. Project
            import numpy as np
            emb_array = np.array(embeddings)
            
            if dimension == "2D":
                _, plot_html = self.visualizer.project_embeddings_2d(emb_array, labels, method.lower())
            else:
                _, plot_html = self.visualizer.project_embeddings_3d(emb_array, labels, method.lower())
                
            # 3. Analyze Quality
            quality_metrics = analyze_cluster_quality(emb_array)
            
            # Format metrics markdown
            if "error" in quality_metrics:
                metrics_md = f"### ⚠️ Quality Analysis Error\n{quality_metrics['error']}"
            else:
                metrics_md = f"""### 📊 Cluster Quality Analysis
- **Quality**: {quality_metrics['quality']}
- **Silhouette Score**: {quality_metrics['silhouette_score']}
- **Davies-Bouldin Score**: {quality_metrics['davies_bouldin_score']}
- **Estimated Clusters**: {quality_metrics['n_clusters']}

*{quality_metrics['interpretation']}*
"""
            
            return f"✅ Generated {dimension} {method} projection for {len(embeddings)} chunks.", plot_html, metrics_md
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return f"❌ Error: {str(e)}", "", ""

    def handle_token_estimation(self, files) -> str:
        """Handle token and cost estimation."""
        total_tokens = 0
        total_cost = 0.0
        details = []
        
        try:
            # 1. Estimate from Files
            if files:
                for f in files:
                    # Simple text extraction for estimation (this is an approximation)
                    try:
                        with open(f.name, "r", encoding="utf-8", errors="ignore") as file:
                            content = file.read()
                            tokens, cost = estimate_tokens(content)
                            total_tokens += tokens
                            total_cost += cost
                            details.append(f"- {os.path.basename(f.name)}: ~{tokens} tokens (${cost:.4f})")
                    except Exception as e:
                        details.append(f"- {os.path.basename(f.name)}: Error reading file")
            
            if total_tokens == 0:
                return "⚠️ No content to estimate."
                
            summary = f"""### 💰 Estimation Result
**Total Tokens**: ~{total_tokens:,}
**Estimated Cost (GPT-4)**: ${total_cost:.4f}

**Breakdown:**
{chr(10).join(details)}
"""
            return summary
            
        except Exception as e:
            logger.error(f"Estimation error: {e}")
            return f"❌ Error during estimation: {str(e)}"

    def handle_graph_generation(self) -> Tuple[str, str, str]:
        """Handle knowledge graph generation."""
        try:
            # 1. Get all documents
            data = self.vector_store.get_all_embeddings()
            if not data or not data.get("documents"):
                return "⚠️ No documents found in collection.", "", ""
            
            documents = data["documents"]
            
            # 2. Build Graph
            self.graph_rag.build_graph(documents)
            
            # 3. Visualize
            graph_path = self.graph_rag.visualize_graph()
            
            # 4. Get Stats
            stats = self.graph_rag.get_graph_stats()
            stats_md = f"""### 🕸️ Graph Statistics
- **Nodes**: {stats['nodes']}
- **Edges**: {stats['edges']}
- **Density**: {stats['density']:.4f}
- **Components**: {stats['components']}
"""
            
            # Read HTML content for display
            if graph_path and os.path.exists(graph_path):
                with open(graph_path, "r", encoding="utf-8") as f:
                    graph_html = f.read()
                # Clean up temp file? Maybe keep it for now or let OS handle it
                # os.remove(graph_path) 
                return f"✅ Graph built with {stats['nodes']} nodes and {stats['edges']} edges.", graph_html, stats_md
            else:
                return "❌ Failed to generate graph visualization.", "", ""
                
        except Exception as e:
            logger.error(f"Graph generation error: {e}")
            return f"❌ Error: {str(e)}", "", ""

    def chat_with_ai(self, message, history, model, rag_technique, database, temperature, max_tokens, top_k, similarity_threshold, chunk_size, chunk_overlap, use_reranker, system_prompt):
        """Chat with AI using RAG."""
        if not message:
            return history, ""
        
        try:
            # Step 1: Retrieve context based on RAG technique
            context = ""
            if rag_technique and rag_technique.lower() not in ["none", ""]:
                try:
                    if "naive" in rag_technique.lower():
                        # Naive RAG: Direct retrieval
                        retrieved_docs = self.vector_store.similarity_search(
                            query=message,
                            k=int(top_k),
                            score_threshold=float(similarity_threshold)
                        )
                        
                    elif "hyde" in rag_technique.lower() or "hypothetical" in rag_technique.lower():
                        # HyDE: Generate hypothetical answer first, use for retrieval
                        logger.info("Using HyDE technique")
                        # Generate hypothesis
                        hypothesis_prompt = f"Generate a hypothetical answer to: {message}"
                        try:
                            if model.lower().startswith("gpt") and self.openai_client:
                                hyp_completion = self.openai_client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": hypothesis_prompt}],
                                    temperature=0.7,
                                    max_tokens=200
                                )
                                hypothesis = hyp_completion.choices[0].message.content
                            else:
                                ollama_hyp = OllamaClient(model=model, temperature=0.7, max_tokens=200)
                                hypothesis = ollama_hyp.chat(hypothesis_prompt)
                            
                            # Use hypothesis for retrieval
                            retrieved_docs = self.vector_store.similarity_search(
                                query=hypothesis,
                                k=int(top_k),
                                score_threshold=float(similarity_threshold)
                            )
                        except Exception as e:
                            logger.error(f"HyDE hypothesis generation failed: {e}, falling back to naive")
                            retrieved_docs = self.vector_store.similarity_search(
                                query=message,
                                k=int(top_k),
                                score_threshold=float(similarity_threshold)
                            )
                    
                    elif "fusion" in rag_technique.lower():
                        # RAG Fusion: Multiple query variations
                        logger.info("Using RAG Fusion technique")
                        query_variations = [
                            message,
                            f"What is {message}?",
                            f"Explain {message}"
                        ]
                        all_docs = []
                        for query in query_variations:
                            docs = self.vector_store.similarity_search(
                                query=query,
                                k=int(top_k) // len(query_variations) + 1,
                                score_threshold=float(similarity_threshold)
                            )
                            all_docs.extend(docs)
                        
                        # Remove duplicates and limit
                        seen = set()
                        retrieved_docs = []
                        for doc in all_docs:
                            if doc.page_content not in seen:
                                seen.add(doc.page_content)
                                retrieved_docs.append(doc)
                                if len(retrieved_docs) >= int(top_k):
                                    break
                    
                    elif "hybrid" in rag_technique.lower():
                        # Hybrid Search: BM25 + Dense
                        logger.info("Using Hybrid Search technique")
                        # Alpha parameter should come from UI (default 0.5)
                        alpha = 0.5  # Will be passed as parameter later
                        retrieved_docs = self.vector_store.hybrid_search(
                            query=message,
                            k=int(top_k),
                            alpha=alpha
                        )
                    
                    elif "graph" in rag_technique.lower() and "hybrid" in rag_technique.lower():
                        # Graph + Vector Hybrid: Combine graph entity search with vector search
                        logger.info("Using Graph + Vector Hybrid technique")
                        
                        # 1. Graph search: Find related entities
                        related_entities = self.graph_rag.graph_search(message, max_hops=2)
                        
                        # 2. Vector search: Standard similarity search
                        vector_docs = self.vector_store.similarity_search(
                            query=message,
                            k=int(top_k),
                            score_threshold=float(similarity_threshold)
                        )
                        
                        # 3. Enhance vector results with entity context
                        retrieved_docs = vector_docs
                        
                        # Add entity context to the response if entities were found
                        if related_entities:
                            entity_context = f"\n\n**Related Entities Found:** {', '.join(related_entities[:10])}"
                            # We'll append this to the context later
                            logger.info(f"Found {len(related_entities)} related entities via graph")
                    
                    else:
                        # Default to naive RAG
                        retrieved_docs = self.vector_store.similarity_search(
                            query=message,
                            k=int(top_k),
                            score_threshold=float(similarity_threshold)
                        )
                    
                    if retrieved_docs:
                        # Format context with source tracking
                        context_parts = []
                        sources = []
                        for i, doc in enumerate(retrieved_docs, 1):
                            # Extract source metadata
                            source_info = doc.metadata.get('source', 'Unknown')
                            # Get just the filename if it's a full path
                            if '/' in source_info or '\\' in source_info:
                                source_info = source_info.split('/')[-1].split('\\')[-1]
                            
                            context_parts.append(f"[Document {i}]\n{doc.page_content}\n")
                            sources.append(f"[{i}] {source_info}")
                        
                        context = "\n".join(context_parts)
                        logger.info(f"Retrieved {len(retrieved_docs)} documents using {rag_technique}")
                    else:
                        logger.info("No documents retrieved")
                        sources = []
                        
                except Exception as e:
                    logger.error(f"Error during retrieval: {e}")
                    context = ""
                    sources = []
            else:
                sources = []
            
            # Step 2: Construct the prompt
            if context:
                full_prompt = f"""{system_prompt}

Context from knowledge base:
{context}

User question: {message}

IMPORTANT: When answering, cite your sources using inline citations like [1], [2], etc. to reference the document numbers above. Use these citations throughout your answer when you reference information from specific documents."""
            else:
                full_prompt = f"{system_prompt}\n\nUser: {message}"
            
            # Step 3: Select and call the appropriate LLM
            response = ""
            if model.lower().startswith("gpt"):
                # Use OpenAI
                if not self.openai_client:
                    response = "❌ OpenAI client not initialized. Please check your API key in .env"
                else:
                    try:
                        completion = self.openai_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ],
                            temperature=float(temperature),
                            max_tokens=int(max_tokens)
                        )
                        response = completion.choices[0].message.content
                    except Exception as e:
                        logger.error(f"OpenAI error: {e}")
                        response = f"❌ OpenAI Error: {str(e)}"
            else:
                # Use Ollama
                try:
                    ollama_client = OllamaClient(
                        model=model,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens)
                    )
                    response = ollama_client.chat(full_prompt)
                except Exception as e:
                    logger.error(f"Ollama error: {e}")
                    response = f"❌ Ollama Error: {str(e)}\n\nMake sure Ollama is running and the model '{model}' is available."
            
            # Step 4: Add source citations if documents were retrieved
            if sources:
                response += f"\n\n---\n**📚 Sources:**\n" + "\n".join(sources)
            
            # Step 5: Update history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, ""
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = f"❌ Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""

    def export_chat(self, history):
        """Export chat history to a Markdown file in Downloads folder."""
        import os
        from datetime import datetime
        from pathlib import Path
        
        if not history:
            return None
        
        # Generate markdown content
        md_lines = [
            "# AI Workdesk Chat Export",
            f"\n**Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n"
        ]
        
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                md_lines.append(f"## 👤 User\n\n{content}\n")
            elif role == "assistant":
                md_lines.append(f"## 🤖 Assistant\n\n{content}\n")
        
        # Save to Downloads folder with timestamp
        downloads_path = Path.home() / "Downloads"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_workdesk_chat_{timestamp}.md"
        file_path = downloads_path / filename
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        
        logger.info(f"Chat exported to {file_path}")
        return str(file_path)


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
                    
                    # Spacer to push content to bottom
                    gr.HTML("<div style='flex-grow: 1; min-height: 200px;'></div>")
                    
                    gr.Markdown("---")
                    
                    # Logout Button
                    with gr.Group():
                         logout_btn = gr.Button("🚪 Logout", variant="secondary", elem_classes=["secondary-btn"])
                    
                    gr.Markdown(self.get_auth_status())
                    
                    # Logo at very bottom (below status)
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
                    with gr.Group(visible=True) as home_page:
                        dashboard_html, refresh_fn = render_dashboard()
                        # Auto-load dashboard data
                        demo.load(fn=refresh_fn, outputs=dashboard_html)

                    # WORKDESK PAGE
                    with gr.Group(visible=False) as workdesk_page:
                        with gr.Tabs():
                            # TAB 1: Embedding LAB
                            with gr.TabItem("🧬 Embedding LAB"):
                                with gr.Tabs():
                                    with gr.TabItem("📤 Ingestion"):
                                        gr.Markdown("### 📄 Document Ingestion")
                                        
                                        with gr.Tabs():
                                            with gr.TabItem("📄 Files"):
                                                file_input = gr.File(
                                                    file_count="multiple", 
                                                    label="Upload Documents (TXT, PDF, MD, DOCX, CSV, JSON, HTML, PPTX, XLSX)"
                                                )
                                                with gr.Row():
                                                    ingest_chunk_size = gr.Dropdown([256, 512, 1024], value=512, label="Chunk Size")
                                                    ingest_chunk_overlap = gr.Slider(0, 200, 50, step=10, label="Overlap")
                                                
                                                chunking_strategy = gr.Dropdown(
                                                    choices=["Fixed Size", "Semantic"],
                                                    value="Fixed Size",
                                                    label="Chunking Strategy"
                                                )
                                                
                                                with gr.Row():
                                                    estimate_btn = gr.Button("💰 Estimate Cost", variant="secondary")
                                                    ingest_btn = gr.Button("🚀 Ingest Documents", variant="primary", elem_classes=["primary-btn"])
                                                
                                                ingest_status = gr.Textbox(label="Status", interactive=False)
                                                
                                                estimate_btn.click(
                                                    self.handle_token_estimation,
                                                    inputs=[file_input],
                                                    outputs=[ingest_status]
                                                )
                                                
                                                ingest_btn.click(
                                                    self.handle_ingestion,
                                                    inputs=[file_input, ingest_chunk_size, ingest_chunk_overlap, chunking_strategy],
                                                    outputs=[ingest_status]
                                                )

                                            with gr.TabItem("🌐 Web"):
                                                gr.Markdown("### 🕸️ Web Crawler")
                                                web_url = gr.Textbox(label="URL", placeholder="https://example.com")
                                                web_depth = gr.Slider(0, 5, value=1, step=1, label="Crawl Depth")
                                                
                                                with gr.Row():
                                                    web_chunk_size = gr.Dropdown([256, 512, 1024], value=512, label="Chunk Size")
                                                    web_chunk_overlap = gr.Slider(0, 200, 50, step=10, label="Overlap")
                                                
                                                web_ingest_btn = gr.Button("🚀 Crawl & Ingest", variant="primary")
                                                web_status = gr.Textbox(label="Status", interactive=False)
                                                
                                                web_ingest_btn.click(
                                                    self.handle_web_ingestion,
                                                    inputs=[web_url, web_depth, web_chunk_size, web_chunk_overlap],
                                                    outputs=[web_status]
                                                )

                                    with gr.TabItem("📋 Metadata"):
                                        gr.Markdown("### 🗄️ Ingested Files")
                                        gr.Markdown("*Showing unique documents (most recent version)*")
                                        
                                        metadata_df = gr.Dataframe(
                                            headers=["ID", "Filename", "Size (bytes)", "Uploaded", "Type"],
                                            label="Document Metadata",
                                            interactive=False,
                                            wrap=True
                                        )
                                        
                                        with gr.Row():
                                            page_slider = gr.Slider(
                                                minimum=1, 
                                                maximum=1, 
                                                step=1, 
                                                value=1, 
                                                label="📄 Page",
                                                scale=2
                                            )
                                            refresh_btn = gr.Button(
                                                "🔄 Refresh", 
                                                variant="secondary",
                                                elem_classes=["secondary-btn"],
                                                scale=1
                                            )
                                        
                                        with gr.Row():
                                            delete_id = gr.Number(
                                                label="🗑️ Entry ID to Delete", 
                                                precision=0,
                                                scale=2
                                            )
                                            delete_btn = gr.Button(
                                                "Delete Entry", 
                                                variant="stop",
                                                elem_classes=["secondary-btn"],
                                                scale=1
                                            )
                                        
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

                                    with gr.TabItem("📚 Collections"):
                                        gr.Markdown("### 🗂️ Collection Management")
                                        gr.Markdown("*Organize documents into separate collections*")
                                        
                                        with gr.Row():
                                            collections_list = gr.Dropdown(
                                                choices=self.vector_store.list_collections(),
                                                value=self.vector_store.collection_name,
                                                label="Active Collection",
                                                allow_custom_value=False
                                            )
                                            refresh_collections_btn = gr.Button("🔄 Refresh", scale=1)
                                        
                                        with gr.Row():
                                            new_collection_name = gr.Textbox(
                                                label="New Collection Name",
                                                placeholder="e.g., legal_docs, research_papers"
                                            )
                                            create_btn = gr.Button("➕ Create", variant="primary")
                                        
                                        with gr.Row():
                                            switch_btn = gr.Button("🔀 Switch to Selected", variant="secondary")
                                            delete_btn_coll = gr.Button("🗑️ Delete Selected", variant="stop")
                                        
                                        collection_status = gr.Textbox(label="Status", interactive=False)
                                        
                                        # Collection callbacks
                                        def refresh_collections():
                                            collections = self.vector_store.list_collections()
                                            return gr.Dropdown(choices=collections)
                                        
                                        def create_collection(name):
                                            if not name:
                                                return "⚠️ Please enter a collection name", gr.Dropdown()
                                            success = self.vector_store.create_collection(name)
                                            if success:
                                                collections = self.vector_store.list_collections()
                                                return f"✅ Created collection: {name}", gr.Dropdown(choices=collections)
                                            return f"❌ Failed to create collection: {name}", gr.Dropdown()
                                        
                                        def switch_collection(name):
                                            if not name:
                                                return "⚠️ Please select a collection"
                                            success = self.vector_store.switch_collection(name)
                                            if success:
                                                return f"✅ Switched to collection: {name}"
                                            return f"❌ Failed to switch to collection: {name}"
                                        
                                        def delete_collection(name):
                                            if not name:
                                                return "⚠️ Please select a collection", gr.Dropdown()
                                            if name == self.vector_store.collection_name:
                                                return "⚠️ Cannot delete active collection", gr.Dropdown()
                                            success = self.vector_store.delete_collection(name)
                                            if success:
                                                collections = self.vector_store.list_collections()
                                                return f"✅ Deleted collection: {name}", gr.Dropdown(choices=collections)
                                            return f"❌ Failed to delete collection: {name}", gr.Dropdown()
                                        
                                        refresh_collections_btn.click(refresh_collections, outputs=[collections_list])
                                        create_btn.click(create_collection, inputs=[new_collection_name], outputs=[collection_status, collections_list])
                                        switch_btn.click(switch_collection, inputs=[collections_list], outputs=[collection_status])
                                        delete_btn_coll.click(delete_collection, inputs=[collections_list], outputs=[collection_status, collections_list])

                                    with gr.TabItem("📊 Visualization"):
                                        gr.Markdown("### 🔮 Embedding Projector")
                                        gr.Markdown("*Visualize your document embeddings in 2D or 3D space*")
                                        
                                        with gr.Row():
                                            with gr.Column(scale=1):
                                                viz_method = gr.Dropdown(
                                                    choices=["UMAP", "t-SNE"], 
                                                    value="UMAP", 
                                                    label="Projection Method"
                                                )
                                                viz_dim = gr.Radio(
                                                    choices=["2D", "3D"], 
                                                    value="2D", 
                                                    label="Dimension"
                                                )
                                                viz_btn = gr.Button("🎨 Generate Projection", variant="primary")
                                                viz_status = gr.Textbox(label="Status", interactive=False)
                                                
                                                # Quality Metrics
                                                quality_md = gr.Markdown("### 📊 Quality Metrics\n*Generate projection to see metrics*")
                                                
                                            with gr.Column(scale=3):
                                                viz_plot = gr.HTML(label="Projection Plot", min_height=600)
                                        
                                        viz_btn.click(
                                            self.handle_visualization,
                                            inputs=[viz_method, viz_dim],
                                            outputs=[viz_status, viz_plot, quality_md]
                                        )

                                    with gr.TabItem("🕸️ Knowledge Graph"):
                                        gr.Markdown("### 🔗 Entity Knowledge Graph")
                                        gr.Markdown("*Visualize relationships between entities (People, Organizations, Locations) in your documents*")
                                        
                                        with gr.Row():
                                            with gr.Column(scale=1):
                                                graph_btn = gr.Button("🕸️ Generate Graph", variant="primary")
                                                graph_status = gr.Textbox(label="Status", interactive=False)
                                                
                                                # Graph Stats
                                                graph_stats_md = gr.Markdown("### 📊 Graph Statistics\n*Generate graph to see stats*")
                                                
                                            with gr.Column(scale=3):
                                                graph_plot = gr.HTML(label="Knowledge Graph", min_height=600)
                                        
                                        graph_btn.click(
                                            self.handle_graph_generation,
                                            inputs=[],
                                            outputs=[graph_status, graph_plot, graph_stats_md]
                                        )

                            # TAB 2: RAG LAB
                            with gr.TabItem("🧠 RAG LAB"):
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
                                            with gr.Column(scale=1):
                                                clear_btn = gr.Button(
                                                    "🗑️ Clear Chat", variant="secondary", elem_classes=["secondary-btn"]
                                                )
                                            with gr.Column(scale=1):
                                                download_btn = gr.Button("📥 Download Chat", variant="secondary", elem_classes=["secondary-btn"])


                                    with gr.Column(scale=2, elem_classes=["glass-panel"]):
                                        provider_dropdown = gr.Dropdown(
                                            choices=["Ollama", "OpenAI"],
                                            value="Ollama",
                                            label="Provider",
                                        )
                                        
                                        model_dropdown = gr.Dropdown(
                                            choices=MODELS["Ollama"],
                                            value="deepseek-r1:7b",
                                            label="Model",
                                            allow_custom_value=True,
                                        )

                                        rag_dropdown = gr.Dropdown(
                                            choices=[
                                                "Naive RAG",
                                                "Hybrid Search",
                                                "Graph + Vector Hybrid",
                                                "HyDE (Hypothetical Document Embeddings)",
                                                "RAG Fusion",
                                                "None",
                                            ],
                                            value="Naive RAG",
                                            label="RAG Technique",
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

                                # Provider change updates model list
                                def update_models(provider):
                                    models = MODELS.get(provider, MODELS["Ollama"])
                                    return gr.update(choices=models, value=models[0])
                                
                                provider_dropdown.change(
                                    update_models,
                                    inputs=[provider_dropdown],
                                    outputs=[model_dropdown]
                                )

                                clear_btn.click(lambda: ([], ""), None, [chatbot, msg])
                                
                                # Download with JavaScript to trigger browser Save As dialog
                                download_btn.click(
                                    None,
                                    inputs=[chatbot],
                                    outputs=None,
                                    js="""(history) => {
                                        if (!history || history.length === 0) {
                                            alert('No chat history to export');
                                            return;
                                        }
                                        
                                        // Generate markdown content
                                        let md = '# AI Workdesk Chat Export\\n\\n';
                                        md += `**Exported**: ${new Date().toLocaleString()}\\n\\n`;
                                        md += '---\\n\\n';
                                        
                                        history.forEach(msg => {
                                            if (msg.role === 'user') {
                                                md += `## 👤 User\\n\\n${msg.content}\\n\\n`;
                                            } else if (msg.role === 'assistant') {
                                                md += `## 🤖 Assistant\\n\\n${msg.content}\\n\\n`;
                                            }
                                        });
                                        
                                        // Create blob and trigger download with Save As dialog
                                        const blob = new Blob([md], {type: 'text/markdown;charset=utf-8'});
                                        const url = URL.createObjectURL(blob);
                                        const a = document.createElement('a');
                                        a.href = url;
                                        
                                        // Generate filename with timestamp
                                        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
                                        a.download = `ai_workdesk_chat_${timestamp}.md`;
                                        
                                        // Trigger download
                                        document.body.appendChild(a);
                                        a.click();
                                        
                                        // Cleanup
                                        setTimeout(() => {
                                            document.body.removeChild(a);
                                            URL.revokeObjectURL(url);
                                        }, 100);
                                    }"""
                                )

                            # TAB 3: Chat LAB (Pure Chat, No RAG)
                            with gr.TabItem("💬 Chat LAB"):
                                with gr.Row(elem_classes=["chat-row"]):
                                    with gr.Column(scale=7):
                                        chatbot_pure = gr.Chatbot(
                                            label="AI Assistant",
                                            show_copy_button=True,
                                            elem_classes=["chat-container"],
                                            type="messages",
                                            height=800,
                                            render_markdown=True,
                                        )

                                        with gr.Row():
                                            msg_pure = gr.Textbox(
                                                label="Message",
                                                placeholder="Ask me anything...",
                                                scale=4,
                                                show_label=False,
                                                container=False,
                                            )
                                            send_btn_pure = gr.Button("Send 🚀", scale=1, variant="primary", elem_classes=["primary-btn"])

                                        with gr.Row():
                                            with gr.Column(scale=1):
                                                clear_btn_pure = gr.Button(
                                                    "🗑️ Clear Chat", variant="secondary", elem_classes=["secondary-btn"]
                                                )
                                            with gr.Column(scale=1):
                                                download_btn_pure = gr.Button("📥 Download Chat", variant="secondary", elem_classes=["secondary-btn"])


                                    with gr.Column(scale=2, elem_classes=["glass-panel"]):
                                        provider_dropdown_pure = gr.Dropdown(
                                            choices=["Ollama", "OpenAI"],
                                            value="Ollama",
                                            label="Provider",
                                        )

                                        model_dropdown_pure = gr.Dropdown(
                                            choices=MODELS["Ollama"],
                                            value="deepseek-r1:7b",
                                            label="Model",
                                            allow_custom_value=True,
                                        )

                                        gr.Markdown("### ⚙️ Model Settings")

                                        temperature_slider_pure = gr.Slider(
                                            minimum=0.0,
                                            maximum=1.0,
                                            value=0.7,
                                            step=0.1,
                                            label="Temperature",
                                        )

                                        max_tokens_slider_pure = gr.Slider(
                                            minimum=100,
                                            maximum=4096,
                                            value=1024,
                                            step=100,
                                            label="Max Tokens",
                                        )

                                        system_prompt_pure = gr.Textbox(
                                            label="System Prompt",
                                            value="You are a helpful AI assistant.",
                                            lines=3,
                                        )

                                        # Dummy inputs for RAG parameters (disabled for pure chat)
                                        rag_none = gr.State("None")
                                        db_none = gr.State("ChromaDB") 
                                        top_k_dummy = gr.State(5)
                                        threshold_dummy = gr.State(0.0)
                                        chunk_dummy = gr.State(512)
                                        overlap_dummy = gr.State(50)
                                        rerank_dummy = gr.State(False)

                                        # Event Wiring for Pure Chat
                                        msg_pure.submit(
                                            self.chat_with_ai,
                                            inputs=[
                                                msg_pure, 
                                                chatbot_pure, 
                                                model_dropdown_pure,
                                                rag_none, # RAG Disabled
                                                db_none,
                                                temperature_slider_pure,
                                                max_tokens_slider_pure,
                                                top_k_dummy,
                                                threshold_dummy,
                                                chunk_dummy,
                                                overlap_dummy,
                                                rerank_dummy,
                                                system_prompt_pure
                                            ],
                                            outputs=[chatbot_pure, msg_pure]
                                        )

                                        send_btn_pure.click(
                                            self.chat_with_ai,
                                            inputs=[
                                                msg_pure, 
                                                chatbot_pure, 
                                                model_dropdown_pure,
                                                rag_none, # RAG Disabled
                                                db_none,
                                                temperature_slider_pure,
                                                max_tokens_slider_pure,
                                                top_k_dummy,
                                                threshold_dummy,
                                                chunk_dummy,
                                                overlap_dummy,
                                                rerank_dummy,
                                                system_prompt_pure
                                            ],
                                            outputs=[chatbot_pure, msg_pure]
                                        )

                                        # Provider change updates model list
                                        provider_dropdown_pure.change(
                                            update_models,
                                            inputs=[provider_dropdown_pure],
                                            outputs=[model_dropdown_pure]
                                        )

                                        clear_btn_pure.click(lambda: None, None, chatbot_pure, queue=False)
                                        download_btn_pure.click(self.export_chat, inputs=[chatbot_pure], outputs=[])


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
        server_port=None, # Allow dynamic port allocation
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


# Features Overview

Comprehensive guide to all AI Workdesk features.

## Smart Dashboard

Your AI-powered command center with intelligent insights.

### Features
- **Live Clock & Weather** - Current time and weather widget
- **Smart Feed** - Timeline combining emails, news, YouTube, and trends
- **Context-Aware Chat** - Ask questions about dashboard items
- **Clickable Items** - All items link to original sources
- **Urgency Scoring** - Items sorted by importance

### How to Use
1. Navigate to **Home** page
2. Review the smart feed for important updates
3. Use the chat bar to ask questions like:
   - "Summarize urgent items"
   - "What are today's trending topics?"
   - "Show me YouTube videos about AI"

---

## Embedding LAB

Process and store documents for RAG queries.

### Ingestion Methods

#### 📄 File Upload
**Supported Formats:**
- Documents: PDF, TXT, MD, DOCX, PPTX
- Data: CSV, JSON, XLSX, XLS
- Web: HTML, HTM
- Images (with OCR): PNG, JPG, TIFF, BMP, GIF

**Steps:**
1. Go to **Embedding LAB** → **Files**
2. Select embedding provider (Ollama/OpenAI/HuggingFace)
3. Choose embedding model
4. Upload files
5. Configure chunking (size: 512, overlap: 50)
6. Click **🚀 Ingest Documents**

#### 🌐 Web Crawling
**Features:**
- Crawl depth: 0-5 levels
- Timeout: 120s (configurable)
- Max pages: 100 (configurable)

**Steps:**
1. Go to **Embedding LAB** → **Web**
2. Enter target URL
3. Set crawl depth (0 = single page, 1 = page + links)
4. Click **🚀 Crawl & Ingest**

#### 🎥 YouTube Ingestion
**Features:**
- Auto-summary generation
- Playlist support
- Timestamp preservation
- Transcript extraction

**Steps:**
1. Go to **Embedding LAB** → **YouTube**
2. Enter YouTube URL(s) (one per line for batch)
3. Enable auto-summary (optional)
4. Enable playlist support (optional)
5. Click **🚀 Process Videos**

---

## RAG LAB

Advanced retrieval-augmented generation with multiple techniques.

### RAG Techniques

#### Naive RAG
**Best for:** Quick queries, simple documents  
**How it works:** Direct similarity search  
**Speed:** ⚡⚡⚡ Fastest

#### HyDE (Hypothetical Document Embeddings)
**Best for:** Complex queries, better semantic matching  
**How it works:** Generates hypothetical answer, then searches  
**Speed:** ⚡⚡ Moderate

#### RAG Fusion
**Best for:** Comprehensive answers  
**How it works:** Multiple query variations, merged results  
**Speed:** ⚡ Slower (multiple searches)

#### Hybrid Search
**Best for:** Balanced semantic + keyword matching  
**How it works:** Combines vector similarity with BM25  
**Speed:** ⚡⚡ Moderate

#### Graph+Vector
**Best for:** Entity-aware retrieval  
**How it works:** Uses knowledge graph + vector search  
**Speed:** ⚡ Slower (graph traversal)

### Configuration Options

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Top-K | 1-20 | 4 | Number of chunks to retrieve |
| Similarity Threshold | 0.0-1.0 | 0.1 | Minimum similarity score |
| Temperature | 0.0-2.0 | 0.7 | AI creativity (0=focused, 2=creative) |
| Max Tokens | 100-8000 | 2000 | Maximum response length |
| Chunk Size | 128-2048 | 512 | Document chunk size |
| Chunk Overlap | 0-512 | 50 | Overlap between chunks |
| Reranker | On/Off | On | Re-rank results for relevance |

### Advanced Settings

**System Prompt:** Customize AI behavior  
**Voice Response:** Enable text-to-speech  
**Provider:** Switch between Ollama and OpenAI

---

## Chat LAB

Direct AI interaction without document retrieval.

### Features
- **No RAG:** Pure chat without document context
- **Multimodal Input:** Upload files for analysis
- **Voice Input:** Record audio queries
- **Export:** Save conversations as Markdown
- **Obsidian Integration:** Export to Obsidian vault

### Use Cases
- General questions
- Code generation
- Creative writing
- File analysis (upload and ask)

---

## Agentic LAB

Multi-agent workflows powered by AutoGen Studio.

### Features
- **Multi-Agent Orchestration:** Create complex agent teams
- **Workflow Management:** Design and run agent workflows
- **Full Screen Mode:** Dedicated interface at `localhost:8081`

### Access
Click **🤖 Agentic LAB** in sidebar to open in new tab.

---

## Visualization

Explore your data visually.

### Embedding Projector

**Dimensions:** 2D or 3D  
**Methods:** PCA, t-SNE, UMAP

**How to Use:**
1. Go to **Embedding LAB** → **Visualization**
2. Select reduction method (UMAP recommended)
3. Choose dimension (3D for exploration, 2D for clarity)
4. Click **🎨 Visualize Embeddings**

### Knowledge Graph

**Features:**
- Entity extraction (People, Organizations, Locations, Products, Events)
- Interactive network visualization
- Dynamic filtering (max nodes, connection strength)
- 2D/3D modes
- Custom background color

**How to Use:**
1. Go to **Embedding LAB** → **Knowledge Graph**
2. Click **🕸️ Generate Graph**
3. Adjust filters:
   - **Max Nodes:** Top connected entities (10-500)
   - **Min Connection Strength:** Filter weak links (1-10)
4. Click **↗️ Open Full Screen** for immersive view

---

## Metadata Management

Track all ingested documents.

### Features
- View all processed files
- See file size, type, upload time
- Pagination support
- Delete entries

### Access
Go to **Embedding LAB** → **Metadata**

---

## Obsidian Integration

Export conversations to your Obsidian vault.

### Setup
Add to `.env`:
```bash
OBSIDIAN_VAULT_PATH=C:\path\to\your\vault
```

### Export Locations
- **YouTube:** `AI Workdesk/YouTube/`
- **Chat:** `AI Workdesk/Chats/`
- **RAG:** `AI Workdesk/RAG/`

### How to Use
Click **💾 Save to Obsidian** button in any LAB.

---

Next: [Configuration Guide](configuration.md)

# Getting Started

Get up and running with AI Workdesk in minutes.

## Prerequisites

- **Python 3.12+** installed
- **UV package manager** ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Ollama** (recommended for local models) - [Download](https://ollama.ai/)
- **Tesseract OCR** (optional, for scanned PDFs) - [Download](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

### 1. Clone and Navigate to Project

First, clone the repository:

```bash
git clone https://github.com/rderakhshan/Work-Desk.git
cd Work-Desk
```

Or if you already have the project, navigate to it:

```bash
cd path/to/Work-Desk
```

### 2. Sync Dependencies

```bash
uv sync
```

This will install all required packages using UV's ultra-fast dependency resolution.

### 3. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys (optional if using Ollama only):

```bash
# For cloud models (optional)
OPENAI_API_KEY=your-openai-api-key-here

# For local models (recommended)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=deepseek-r1:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### 4. Install Ollama Models

Download recommended models for local use:

```bash
ollama pull deepseek-r1:7b      # Chat model with reasoning
ollama pull nomic-embed-text    # Embedding model
```

### 5. Launch the Application

```bash
uv run ai-workdesk-ui
```

The application will start at `http://localhost:7860`

## First Login

**Default Credentials:**
- Username: `admin`
- Password: `admin123`

**Alternative:**
- Username: `demo`
- Password: `demo123`

> **Tip:** You can change credentials in `src/ai_workdesk/ui/gradio_app.py`

## Your First RAG Query

### Step 1: Ingest a Document

1. Navigate to **Embedding LAB** → **Files**
2. Select embedding provider: **Ollama**
3. Choose model: **nomic-embed-text**
4. Upload a PDF or text file
5. Click **🚀 Ingest Documents**

### Step 2: Query Your Data

1. Go to **RAG LAB**
2. Select RAG technique: **Naive RAG** (simplest)
3. Type your question in the chat box
4. Press Enter to get AI-powered answers with source citations!

## Next Steps

- [Explore Features](features.md) - Learn about all capabilities
- [Configure Settings](configuration.md) - Customize your setup
- [Advanced RAG](advanced.md) - Master retrieval techniques

## Quick Tips

- **Temperature:** 0 = focused, 2 = creative
- **Top-K:** Number of document chunks to retrieve (default: 4)
- **Similarity Threshold:** Lower = more permissive (recommended: 0.1-0.3)
- **Chunk Size:** Larger = more context, smaller = more precise (default: 512)

---

Need help? Check the [Troubleshooting](troubleshooting.md) guide.

# 🚀 AI Workdesk

A professional Python workdesk for developing and utilizing various AI tools. Built with modern best practices using `uv` for ultra-fast dependency management.

## ✨ Key Features

- 🏠 **Smart Homepage**: Full-width glassmorphism dashboard with AI-powered insights
- ⚙️ **Granular Control**: Advanced settings for Top-K, Chunk Size, Overlap, and Reranking
- 🧪 **Engineering Labs**: Dedicated tabs for **Embedding** (Ingestion) and **Chat** (Retrieval)
- 🗄️ **Multi-DB Support**: Integration with ChromaDB, FAISS, PGVector, and more
- 🌐 **Flexible Providers**: Support for OpenAI, HuggingFace, Ollama, and Google Gemini embeddings
- 💬 **Context-Aware Chat**: Chat with your dashboard items using selected AI models

## 🆕 Recent Updates (November 22, 2025)

### 🏠 Smart Homepage ("Project Ambitions")
- ✅ **Full-Width Glassmorphism UI**: Modern, asymmetrical 65/35 split layout
- ✅ **Time & Weather Widget**: Live clock and weather display at top right
- ✅ **Smart Feed**: Timeline-style feed combining emails, news, videos, and trends
- ✅ **Clickable Items**: All dashboard items link to their original sources
- ✅ **Context-Aware AI Chat**: Chat with dashboard items using Ollama or OpenAI
- ✅ **Default Model**: DeepSeek-R1:7b for intelligent reasoning
- ✅ **Floating Stats**: Real-time urgency scores and item counts
- ✅ **Quick Actions**: One-click access to common tasks

### RAG Enhancements

### Vector Store Improvements
- ✅ **Collection Stats**: Real-time document count logging
- ✅ **Better Error Messages**: Shows actual distances when retrieval fails
- ✅ **Threshold Guidance**: Recommends optimal similarity threshold values (0.3-0.4 for ChromaDB)

## 📁 Project Structure

```
ai-workdesk/
├── src/
│   └── ai_workdesk/           # Main package
│       ├── core/              # Core utilities
│       │   ├── auth.py       # Authentication manager
│       │   ├── config.py     # Configuration management
│       │   └── logger.py     # Logging setup
│       ├── rag/              # RAG pipeline
│       │   ├── ingestion.py  # Document processing & chunking
│       │   ├── vector_store.py # Vector database management
│       │   └── metadata_store.py # Document metadata tracking
│       ├── smart_dashboard/  # Smart Homepage
│       │   ├── ui.py         # Dashboard UI components
│       │   ├── data_engine.py # Data aggregation
│       │   ├── ai_processor.py # AI chat integration
│       │   ├── models.py     # Data models
│       │   └── fetchers/     # Data source fetchers
│       │       ├── email_fetcher.py
│       │       ├── rss_fetcher.py
│       │       ├── youtube_fetcher.py
│       │       └── trends_fetcher.py
│       ├── ui/               # User interfaces
│       │   ├── gradio_app.py # Gradio web interface
│       │   └── assets/       # UI assets (logo, etc.)
│       ├── tools/            # AI tools collection
│       │   ├── llm/         # LLM tools
│       │   │   └── ollama_client.py # Ollama local model wrapper
│       │   ├── embeddings/  # Embedding tools
│       │   ├── retrieval/   # RAG tools
│       │   └── vision/      # Vision tools
│       └── utils/           # Shared utilities
├── docs/                    # Documentation
├── tests/                   # Test suite
├── data/                    # Data directory
│   └── chroma_db/          # ChromaDB persistent storage
└── config/                  # Configuration files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- `uv` installed ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Recommended**: [Ollama](https://ollama.ai/) for local model support (privacy and offline use)

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd "C:\\RAG LAB"
```

2. **Sync dependencies:**
```bash
uv sync
```

3. **Set up environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys (optional for Ollama-only usage)
```

4. **Run the Web UI:**
```bash
uv run ai-workdesk-ui
```

**Default Login Credentials:**
- Username: `admin`
- Password: `admin123`

5. **Install Ollama for Local Models (Recommended):**
```bash
# Download and install Ollama from https://ollama.ai/

# Pull recommended models
ollama pull deepseek-r1:7b      # Chat model (reasoning) - Default for Smart Homepage
ollama pull nomic-embed-text    # Embedding model
```

## 🏠 Smart Homepage

The Smart Homepage is your AI-powered command center, providing an intelligent overview of your digital workspace.

### Features

**🎨 Full-Width Glassmorphism Design**
- Modern asymmetrical layout (65/35 split)
- Borderless glass components with high blur effects
- Editorial-style typography using Outfit font
- Smooth animations and hover effects

**⏰ Time & Weather Widget**
- Live clock display
- Current date
- Weather icon (customizable with real API integration)
- Located at top right for quick reference

**📊 Smart Feed**
- Timeline-style feed combining multiple sources:
  - 📧 Email updates
  - 📰 RSS news feeds
  - 📺 YouTube videos
  - 📈 Trending topics
- All items are **clickable** and link to their original sources
- Sorted by timestamp for latest updates first

**💬 Context-Aware AI Chat**
- Chat bar positioned at the top for easy access
- **Default Model**: DeepSeek-R1:7b (Ollama)
- **Provider Selection**: Switch between Ollama and OpenAI
- **Full Context**: AI receives complete details of all dashboard items
  - Titles, summaries, links, urgency scores, timestamps
- Ask questions like:
  - "Summarize the urgent items"
  - "What are the trending topics today?"
  - "Show me YouTube videos about AI"

**📈 Floating Stats**
- Total items count
- Critical actions (urgency > 60)
- Real-time updates

**⚡ Quick Actions**
- One-click buttons for common tasks
- "Clear Inbox" and "Start Focus" modes

### Customization

The Smart Homepage can be extended with additional data sources by creating new fetchers in `src/ai_workdesk/smart_dashboard/fetchers/`.

## 🔐 Authentication Setup

### Option 1: Environment Variables (Recommended)

Create a `.env` file in the project root:

```bash
# LLM API Keys (Optional - for cloud models)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Model Configuration
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.7

# Ollama Configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=deepseek-r1:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Option 2: Ollama Local Models (Privacy-First)

For complete privacy and offline usage, use Ollama without any API keys:

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)
2. **Pull Models**:
   ```bash
   ollama pull deepseek-r1:7b     # Reasoning chat model
   ollama pull nomic-embed-text   # Embedding model
   ```
3. **Configure `.env`**:
   ```bash
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_CHAT_MODEL=deepseek-r1:7b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   ```

The system will automatically use Ollama models when configured, with no cloud API keys required.

### Check Authentication Status

```bash
uv run python -c "from ai_workdesk import get_auth_manager; get_auth_manager().display_authentication_status()"
```

## 📄 Document Ingestion & RAG Pipeline

The AI Workdesk includes a complete RAG (Retrieval-Augmented Generation) pipeline with document ingestion and vector storage capabilities.

### 🧬 Embedding LAB (Document Ingestion)

Upload and process documents to build your knowledge base:

1. **Navigate to Work Desk → Embedding LAB**
2. **Upload Documents**: Support for multiple formats (see below)
3. **Configure Ingestion Settings**:
   - **Chunk Size**: 256, 512, 1024, or 2048 tokens
   - **Chunk Overlap**: 0-200 tokens (recommended: 50)
   - **Embedding Model**: Choose from:
     - **Ollama** (default, privacy-first, offline)
     - OpenAI
     - HuggingFace
     - Google Gemini
4. **Click "Ingest & Embed"** to process and store documents

The system will:
- Load and parse your documents
- Split them into optimized chunks
- Generate embeddings using the selected model
- Store vectors in ChromaDB for fast retrieval

### 💬 Chat LAB (RAG-Enhanced Chat)

Interact with your documents using advanced RAG techniques:

**Provider-Based Model Selection:**
- **Provider Dropdown**: Choose between OpenAI or Ollama
- **Dynamic Model Filtering**: Model list updates based on selected provider
  - **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo
  - **Ollama**: deepseek-r1:7b, gemma3:4b, llama3, mistral, phi3

**RAG Techniques** (Choose your retrieval strategy):
- **Naive RAG**: Direct similarity search (fastest, good for most cases)
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothesis first for better semantic matching
- **RAG Fusion**: Uses multiple query variations and merges results for comprehensive retrieval
- **None**: Chat without document retrieval

**Chat Features:**
- **Source Citations**: Automatically shows which documents were used in the response
- **Export Chat**: Download conversation history as Markdown with browser "Save As" dialog
- **Temperature**: 0.0 (focused) to 2.0 (creative)
- **Max Tokens**: Up to 8192 tokens

**Advanced RAG Settings:**
- **Top-K Retrieval**: Number of chunks to retrieve (1-20)
- **Similarity Threshold**: Minimum similarity score (0.0-1.0)
  - **Recommended**: 0.3-0.4 for ChromaDB (uses distance metric)
  - Lower values = more permissive retrieval
- **Chunk Size**: Token size for document chunks
- **Chunk Overlap**: Overlap between consecutive chunks
- **Reranker**: Enable/disable result reranking
- **System Prompt**: Custom instructions for the AI

### Supported File Formats

- **Text Files** (`.txt`): Plain text documents
- **PDF Files** (`.pdf`): Portable Document Format
- **Markdown** (`.md`): Markdown formatted documents
- **Word Documents** (`.docx`): Microsoft Word files
- **CSV Files** (`.csv`): Comma-separated values
- **JSON Files** (`.json`): JSON data files
- **HTML Files** (`.html`, `.htm`): Web pages
- **PowerPoint** (`.pptx`): Microsoft PowerPoint presentations
- **Excel Files** (`.xlsx`, `.xls`): Microsoft Excel spreadsheets

### Vector Store

- **Default**: ChromaDB (persistent storage)
- **Location**: `./data/chroma_db/` directory
- **Embedding Models**: 
  - **Ollama**: `nomic-embed-text` (default, local)
  - **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2`
  - **OpenAI**: `text-embedding-3-small`
- **Features**: Automatic persistence, metadata support, similarity search

## 📚 Usage Examples

### Using the Smart Homepage Chat

```python
# The Smart Homepage automatically provides context to the AI
# Just type your question in the chat bar at the top:

# Example queries:
"What are my most urgent items?"
"Summarize the trending topics"
"Show me all YouTube videos from today"
"What emails need immediate attention?"
```

### Using Ollama Client (Local, Privacy-First)

```python
from ai_workdesk.tools.llm.ollama_client import OllamaClient

# Initialize with default model from .env (deepseek-r1:7b)
client = OllamaClient()

# Chat with the model
response = client.chat("Explain quantum computing in simple terms")
print(response)

# Use a different model dynamically
response = client.chat("Hello!", model="llama3")
print(response)

# List available models
models = client.list_models()
print(f"Available models: {models}")
```

### Using OpenAI (Cloud)

```python
from ai_workdesk.core import get_settings
from openai import OpenAI

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

response = client.chat.completions.create(
    model=settings.default_llm_model,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## 🔧 Configuration

All configuration is managed through environment variables and `pyproject.toml`:

### Environment Variables

See `.env.example` for all available options:

**LLM & Model Settings:**
- **Cloud API Keys**: OpenAI, Anthropic, Google, Cohere (optional)
- **Ollama Settings**: Base URL, chat model, embedding model
- **Model Defaults**: Default models, temperature, max tokens

**Ollama Configuration (Local Models):**
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=deepseek-r1:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

**Application Settings:**
- Environment, log level, paths
- Vector Store: ChromaDB configuration
- Security: Rate limiting, timeouts, retries
- Advanced RAG: Top-K retrieval, chunk size/overlap, reranking

## 📦 Installing Optional Dependencies

The project uses optional dependency groups for different AI tools:

```bash
# Install LLM tools (OpenAI, Anthropic, LangChain, etc.)
uv sync --extra llm

# Install embedding tools
uv sync --extra embeddings

# Install vision tools
uv sync --extra vision

# Install everything
uv sync --extra all

# Install dev dependencies (already installed by default)
uv sync --extra dev
```

## 💻 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ai_workdesk --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_config.py
```

### Code Quality

```bash
# Format code with ruff
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking with mypy
uv run mypy src/
```

## 📖 Available Dependency Groups

| Group | Description | Install Command |
|-------|-------------|-----------------|
| `core` | Base dependencies (always installed) | `uv sync` |
| `llm` | LLM tools (OpenAI, Anthropic, LangChain) | `uv sync --extra llm` |
| `embeddings` | Embedding & vector stores | `uv sync --extra embeddings` |
| `vision` | Computer vision tools | `uv sync --extra vision` |
| `all` | All AI tools | `uv sync --extra all` |
| `dev` | Development tools | Installed by default |

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting: `uv run pytest && uv run ruff check .`
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [uv](https://github.com/astral-sh/uv) - An extremely fast Python package installer
- Local model support powered by [Ollama](https://ollama.ai/) - Run LLMs locally
- RAG pipeline built with [LangChain](https://www.langchain.com/) - Framework for LLM applications
- Uses [Pydantic](https://docs.pydantic.dev/) for settings management
- Logging powered by [Loguru](https://github.com/Delgan/loguru)
- Beautiful terminal output with [Rich](https://github.com/Textualize/rich)
- Web UI built with [Gradio](https://www.gradio.app/) - Fast web interfaces for ML

---

**Happy Building! 🚀**

For questions or issues, please open an issue on the repository.

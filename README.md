# 🚀 AI Workdesk

A professional Python workdesk for developing and utilizing various AI tools. Built with modern best practices using `uv` for ultra-fast dependency management.

- ⚙️ **Granular Control**: Advanced settings for Top-K, Chunk Size, Overlap, and Reranking
- 🧪 **Engineering Labs**: Dedicated tabs for **Embedding** (Ingestion) and **Chat** (Retrieval)
- 🗄️ **Multi-DB Support**: Integration with ChromaDB, FAISS, PGVector, and more
- 🌐 **Flexible Providers**: Support for OpenAI, HuggingFace, Ollama, and Google Gemini embeddings

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
│       │   └── vector_store.py # Vector database management
│       ├── ui/               # User interfaces
│       │   └── gradio_app.py # Gradio web interface
│       ├── tools/            # AI tools collection
│       │   ├── llm/         # LLM tools
│       │   │   └── ollama_client.py # Ollama local model wrapper
│       │   ├── embeddings/  # Embedding tools
│       │   ├── retrieval/   # RAG tools
│       │   └── vision/      # Vision tools
│       └── utils/           # Shared utilities
├── docs/                    # Documentation
│   ├── IMPLEMENTATION_PLAN_OLLAMA.md # Ollama integration plan
│   ├── QUICK_START.md      # Quick start guide
│   ├── UI_GUIDE.md         # UI usage guide
│   └── system_prompt_template.md # System prompt examples
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── manual/             # Manual test scripts
├── notebooks/               # Jupyter notebooks
├── data/                    # Data directory
│   └── chroma_db/          # ChromaDB persistent storage
├── config/                  # Configuration files
└── scripts/                 # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- `uv` installed ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Optional**: [Ollama](https://ollama.ai/) for local model support (recommended for privacy and offline use)

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

# Edit .env and add your API keys
```

4. **Run the Web UI:**
```bash
uv run ai-workdesk-ui
```

**Default Login Credentials:**
- Username: `admin`
- Password: `admin123`

5. **Optional - Install Ollama for Local Models:**
```bash
# Download and install Ollama from https://ollama.ai/

# Pull recommended models
ollama pull deepseek-r1:7b      # Chat model (reasoning)
ollama pull nomic-embed-text   # Embedding model
```

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

### Option 3: System Keyring (Advanced)

For enhanced security with cloud APIs, use UV's keyring provider:

```bash
# Enable keyring provider
$env:UV_KEYRING_PROVIDER="subprocess"

# Or add to pyproject.toml:
# [tool.uv]
# keyring-provider = "subprocess"
```

### Check Authentication Status

```bash
uv run python -c "from ai_workdesk import get_auth_manager; get_auth_manager().display_authentication_status()"
```

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

### Interactive Development

```bash
# Start IPython shell with project loaded
uv run ipython

# Start Jupyter notebook
uv run jupyter notebook

# Start Jupyter lab
uv run jupyter lab
```

## 📄 Document Ingestion & RAG Pipeline

The AI Workdesk includes a complete RAG (Retrieval-Augmented Generation) pipeline with document ingestion and vector storage capabilities.

### 🧬 Embedding LAB (Document Ingestion)

Upload and process documents to build your knowledge base:

1. **Navigate to Work Desk → Embedding LAB**
2. **Upload Documents**: Support for `.txt`, `.pdf`, and `.md` files
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

**Basic Settings:**
- **Model Selection**: 
  - **Cloud**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
  - **Local (Ollama)**: DeepSeek-R1:7b, Llama 3, Mistral, and more
- **RAG Technique**: Naive RAG, Hybrid Search, Contextual RAG, Graph RAG
- **Embedding Model**: Ollama (default), OpenAI, HuggingFace, Google Gemini
- **Database**: ChromaDB, FAISS, PGVector, SQLite, Pinecone
- **Temperature**: 0.0 (focused) to 2.0 (creative)
- **Max Tokens**: Up to 8192 tokens

**Advanced RAG Settings:**
- **Top-K Retrieval**: Number of chunks to retrieve (1-20)
- **Similarity Threshold**: Minimum similarity score (0.0-1.0)
- **Chunk Size**: Token size for document chunks
- **Chunk Overlap**: Overlap between consecutive chunks
- **Reranker**: Enable/disable result reranking
- **System Prompt**: Custom instructions for the AI

### Supported File Formats

- **Text Files** (`.txt`): Plain text documents
- **PDF Files** (`.pdf`): Portable Document Format
- **Markdown** (`.md`): Markdown formatted documents

### Vector Store

- **Default**: ChromaDB (persistent storage)
- **Location**: `./data/chroma_db/` directory
- **Embedding Models**: 
  - **Ollama**: `nomic-embed-text` (default, local)
  - **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2`
  - **OpenAI**: `text-embedding-3-small`
- **Features**: Automatic persistence, metadata support, similarity search


## 📚 Usage Examples

### Basic Usage

```python
from ai_workdesk import get_settings, get_auth_manager, get_logger

# Get settings
settings = get_settings()
print(f"Environment: {settings.environment}")
print(f"Default model: {settings.default_llm_model}")

# Check authentication
auth = get_auth_manager()
auth.display_authentication_status()

# Use logger
logger = get_logger(__name__)
logger.info("AI Workdesk initialized!")
```

### Using LLM Tools

**Option 1: Using Ollama (Local, Privacy-First)**

```python
from ai_workdesk.tools.llm.ollama_client import OllamaClient

# Initialize with default model from .env (deepseek-r1:7b)
client = OllamaClient()

# Chat with the model
response = client.chat("Explain quantum computing in simple terms")
print(response)

# Use a different model
client = OllamaClient(model="llama3")
response = client.chat("Hello!")
print(response)
```

**Option 2: Using OpenAI (Cloud)**

```bash
# First install LLM extras
uv sync --extra llm
```

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

### Project Settings

Edit `pyproject.toml` for:
- Dependencies
- Development tools configuration (ruff, pytest, mypy)
- Package metadata

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

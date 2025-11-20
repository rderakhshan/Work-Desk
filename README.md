# 🚀 AI Workdesk

A professional Python workdesk for developing and utilizing various AI tools. Built with modern best practices using `uv` for ultra-fast dependency management.

## ✨ Features

- 🏗️ **Modern Project Structure**: Follows Python src layout best practices
- ⚡ **Ultra-Fast Setup**: Powered by `uv` - 10-100x faster than pip
- 🔐 **Secure Authentication**: Environment-based API key management
- 📦 **Modular Design**: Organized tools for LLMs, embeddings, RAG, and vision
- 🧪 **Ready for Testing**: Pre-configured pytest, ruff, and mypy
- 📝 **Type-Safe**: Full type hints with mypy validation
- 🎨 **Beautiful Console**: Rich terminal output with loguru logging

## 📁 Project Structure

```
ai-workdesk/
├── src/
│   └── ai_workdesk/           # Main package
│       ├── core/              # Core utilities
│       │   ├── auth.py       # Authentication manager
│       │   ├── config.py     # Configuration management
│       │   └── logger.py     # Logging setup
│       ├── tools/            # AI tools collection
│       │   ├── llm/         # LLM tools
│       │   ├── embeddings/  # Embedding tools
│       │   ├── retrieval/   # RAG tools
│       │   └── vision/      # Vision tools
│       └── utils/           # Shared utilities
├── tests/                   # Test suite
├── notebooks/               # Jupyter notebooks
├── data/                    # Data directory
├── config/                  # Configuration files
└── scripts/                 # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- `uv` installed ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd "c:\Users\Riemann\Documents\AI Engineering LAB\RAG\RAG LAB"
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

4. **Test the installation:**
```bash
uv run ai-workdesk
```

## 🔐 Authentication Setup

### Option 1: Environment Variables (Recommended)

Create a `.env` file in the project root:

```bash
# LLM API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Model Configuration
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.7
```

### Option 2: System Keyring (Advanced)

For enhanced security, use UV's keyring provider:

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

### Using LLM Tools (After installing `llm` extras)

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

- **LLM API Keys**: OpenAI, Anthropic, Google, Cohere
- **Model Settings**: Default models, temperature, max tokens
- **Application Settings**: Environment, log level, paths
- **Vector Store**: ChromaDB configuration
- **Security**: Rate limiting, timeouts, retries

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
- Uses [Pydantic](https://docs.pydantic.dev/) for settings management
- Logging powered by [Loguru](https://github.com/Delgan/loguru)
- Beautiful terminal output with [Rich](https://github.com/Textualize/rich)

---

**Happy Building! 🚀**

For questions or issues, please open an issue on the repository.

# 🚀 AI Workdesk - Quick Reference Guide

## 🎯 Common Commands

### Project Setup
```bash
# Sync all dependencies
uv sync

# Install with LLM tools
uv sync --extra llm

# Install everything
uv sync --extra all
```

### Running Code
```bash
# Run the main CLI
uv run ai-workdesk

# Run authentication setup
uv run python scripts/setup_auth.py

# Run any Python script
uv run python your_script.py

# Open IPython shell
uv run ipython

# Start Jupyter Lab
uv run jupyter lab
```

### Testing & Quality
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/unit/test_config.py

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Type checking
uv run mypy src/
```

### Dependency Management
```bash
# Add a new package
uv add package-name

# Add to dev dependencies
uv add --dev package-name

# Remove a package
uv remove package-name

# Update dependencies
uv sync --upgrade
```

## 📦 Project Components

### Core Modules
- `ai_workdesk.core.config` - Configuration & settings
- `ai_workdesk.core.auth` - Authentication management
- `ai_workdesk.core.logger` - Logging setup

### Import Examples
```python
from ai_workdesk import (
    get_settings,      # Get configuration
    get_auth_manager,  # Get auth manager
    get_logger,        # Get logger
)

# Or import directly
from ai_workdesk.core import Settings
from ai_workdesk.core.auth import AuthenticationManager
```

## 🔐 Environment Variables

Edit `.env` file for configuration:

```bash
# Required for using AI services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional model settings
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.7
LOG_LEVEL=INFO
```

## 📁 Directory Layout

```
.
├── src/ai_workdesk/    # Your code goes here
│   ├── core/           # Core utilities (don't modify often)
│   ├── tools/          # Add your AI tools here
│   │   ├── llm/       # LLM-related tools
│   │   ├── embeddings/# Embedding tools
│   │   ├── retrieval/ # RAG tools
│   │   └── vision/    # Vision tools
│   └── utils/          # Shared utilities
│
├── tests/              # Your tests go here
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
│
├── notebooks/          # Jupyter notebooks for experiments
├── data/              # Data files
│   ├── raw/           # Original data
│   ├── processed/     # Processed data
│   └── outputs/       # Generated outputs
│
├── scripts/           # Utility scripts
├── config/            # Configuration files
└── logs/              # Application logs
```

## ✨ Quick Start Code Snippets

### Check Configuration
```python
from ai_workdesk import get_settings

settings = get_settings()
print(f"Model: {settings.default_llm_model}")
print(f"Environment: {settings.environment}")
```

### Check Authentication
```python
from ai_workdesk import get_auth_manager

auth = get_auth_manager()
auth.display_authentication_status()

# Check specific service
if auth.validate_service("openai"):
    print("OpenAI is configured!")
```

### Use Logging
```python
from ai_workdesk import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.debug("Debug information")
logger.error("Error occurred")
```

### Create Your First Tool (After `uv sync --extra llm`)
```python
# src/ai_workdesk/tools/llm/simple_chat.py
from openai import OpenAI
from ai_workdesk.core import get_settings, get_logger

logger = get_logger(__name__)

def chat(message: str) -> str:
    """Simple chat with OpenAI."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    logger.info(f"Sending message to {settings.default_llm_model}")
    
    response = client.chat.completions.create(
        model=settings.default_llm_model,
        messages=[{"role": "user", "content": message}],
    )
    
    return response.choices[0].message.content

# Usage:
# result = chat("Hello! What is AI?")
# print(result)
```

## 🎓 Tips & Tricks

### 1. Always Activate Virtual Environment
```bash
# UV automatically uses .venv, just use:
uv run <command>
```

### 2. Keep Dependencies Organized
- Core deps → `dependencies` in pyproject.toml
- Optional deps → `[project.optional-dependencies]`
- Dev deps → `[tool.uv.dev-dependencies]`

### 3. Use Type Hints
```python
def process_text(text: str, max_length: int = 100) -> str:
    """Type hints help with IDE autocomplete and mypy checks."""
    return text[:max_length]
```

### 4. Environment-Specific Settings
```bash
# Development
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Production
ENVIRONMENT=production
LOG_LEVEL=WARNING
```

### 5. Test Before Committing
```bash
# Run these before git commit
uv run ruff format .
uv run ruff check --fix .
uv run pytest
```

## 🔗 Useful Links

- [UV Documentation](https://docs.astral.sh/uv/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

## 📞 Getting Help

1. Check [README.md](README.md) for detailed documentation
2. Look at [walkthrough.md](.gemini/antigravity/brain/*/walkthrough.md) for examples
3. Review test files in `tests/` for usage patterns
4. Check configuration in `.env.example`

---

**Last Updated**: 2025-11-20  
**Project Version**: 0.1.0

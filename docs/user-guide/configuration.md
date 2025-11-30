# Configuration Guide

Complete configuration reference for AI Workdesk.

## Environment Variables

AI Workdesk uses a `.env` file for configuration. Copy `.env.example` to `.env` and configure the following variables:

### Required Settings

```bash
# Application Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
LOG_FILE=logs/ai_workdesk.log
```

### LLM Provider Configuration

#### Ollama (Recommended for Local Use)

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=deepseek-r1:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

**Available Ollama Models:**
- **Chat Models:** `deepseek-r1:7b`, `llama3.2`, `mistral`, `qwen2.5`
- **Embedding Models:** `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`

#### OpenAI

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

**Supported Models:**
- **Chat:** `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Embeddings:** `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`

#### Anthropic (Coming Soon)

```bash
ANTHROPIC_API_KEY=your-api-key-here
```

#### Google AI (Coming Soon)

```bash
GOOGLE_API_KEY=your-api-key-here
```

#### Cohere

```bash
COHERE_API_KEY=your-api-key-here
```

### Vector Store Configuration

```bash
# ChromaDB (Default)
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=ai_workdesk_docs

# Default Embedding Provider
DEFAULT_EMBEDDING_PROVIDER=ollama  # Options: ollama, openai, huggingface
```

### Obsidian Integration

```bash
# Path to your Obsidian vault (optional)
OBSIDIAN_VAULT_PATH=C:\path\to\your\vault
```

### Authentication

Edit `src/ai_workdesk/ui/gradio_app.py` to configure users:

```python
demo.launch(
    auth=[
        ("admin", "admin123"),
        ("demo", "demo123")
    ],
    server_name="0.0.0.0",
    server_port=7860
)
```

## Model Configuration

### Default Model Settings

Located in `src/ai_workdesk/core/config.py`:

```python
class Settings:
    default_llm_model: str = "deepseek-r1:7b"
    default_embedding_provider: str = "ollama"
    default_temperature: float = 0.7
    max_tokens: int = 2000
```

### Changing Default Models

**Option 1: Environment Variables**
```bash
DEFAULT_LLM_MODEL=llama3.2
DEFAULT_EMBEDDING_PROVIDER=openai
```

**Option 2: Edit config.py**
```python
default_llm_model: str = "gpt-4"
default_embedding_provider: str = "openai"
```

## RAG Configuration

### Chunking Parameters

Configure in the Embedding LAB UI or edit defaults:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Chunk Size | 512 | 128-2048 | Size of document chunks in characters |
| Chunk Overlap | 50 | 0-512 | Overlap between consecutive chunks |
| Separator | `\n\n` | Any string | Text separator for chunking |

### Retrieval Parameters

Configure in RAG LAB UI:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Top-K | 4 | 1-20 | Number of chunks to retrieve |
| Similarity Threshold | 0.1 | 0.0-1.0 | Minimum similarity score (lower = more permissive) |
| Temperature | 0.7 | 0.0-2.0 | LLM creativity (0=focused, 2=creative) |
| Max Tokens | 2000 | 100-8000 | Maximum response length |
| Use Reranker | On | On/Off | Re-rank results for relevance |

## Advanced Settings

### OCR Configuration

For scanned PDFs and images, install Tesseract OCR:

**Windows:**
```bash
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

**Linux/Mac:**
```bash
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract              # macOS
```

### Knowledge Graph Settings

Configure entity extraction in `src/ai_workdesk/rag/graph_rag.py`:

```python
# Entity types to extract
ENTITY_TYPES = ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]

# Minimum entity frequency
MIN_ENTITY_FREQUENCY = 2

# Minimum relationship strength
MIN_RELATIONSHIP_STRENGTH = 0.3
```

### Web Crawling Limits

Edit in `src/ai_workdesk/ui/tabs/workdesk/ingestion.py`:

```python
max_pages = gr.Slider(
    minimum=1,
    maximum=100,  # Adjust maximum pages
    value=10,
    step=1,
    label="Max Pages"
)
```

## Performance Tuning

### Memory Optimization

For large document collections:

```python
# In vector_store.py
BATCH_SIZE = 100  # Process documents in batches
MAX_CACHE_SIZE = 1000  # Limit cached embeddings
```

### GPU Acceleration

For Ollama with GPU:

```bash
# NVIDIA GPU
OLLAMA_GPU=1

# AMD GPU
OLLAMA_GPU=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### Database Optimization

ChromaDB settings in `src/ai_workdesk/rag/vector_store.py`:

```python
client_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
    anonymized_telemetry=False
)
```

## Logging Configuration

### Log Levels

Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

```bash
LOG_LEVEL=INFO  # Change to DEBUG for verbose logging
```

### Log File Location

```bash
LOG_FILE=logs/ai_workdesk.log
```

### Viewing Logs

```bash
# Tail logs in real-time
tail -f logs/ai_workdesk.log

# Windows PowerShell
Get-Content logs/ai_workdesk.log -Wait -Tail 50
```

## Troubleshooting Configuration

### Common Issues

**Issue: "Model not found"**
```bash
# Verify Ollama is running
ollama list

# Pull missing model
ollama pull deepseek-r1:7b
```

**Issue: "ChromaDB connection error"**
```bash
# Check permissions
chmod -R 755 ./chroma_db

# Reset database
rm -rf ./chroma_db
```

**Issue: "API key invalid"**
```bash
# Verify .env file exists
ls -la .env

# Check key format (no quotes)
OPENAI_API_KEY=sk-...
```

## Security Best Practices

1. **Never commit `.env` file** - Already in `.gitignore`
2. **Use strong passwords** - Change default auth credentials
3. **Restrict network access** - Use `server_name="127.0.0.1"` for local-only
4. **Rotate API keys** - Regularly update cloud provider keys
5. **Enable HTTPS** - Use reverse proxy (nginx/caddy) for production

## Next Steps

- [Advanced Topics](advanced.md) - RAG techniques and optimization
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [API Reference](api-reference.md) - Developer documentation

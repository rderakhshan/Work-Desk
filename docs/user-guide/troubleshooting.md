# Troubleshooting

Common issues and solutions for AI Workdesk.

## Installation Issues

### UV Installation Fails

**Symptoms:**
```
Command 'uv' not found
```

**Solution:**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### Dependency Conflicts

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement
```

**Solution:**
```bash
# Clear UV cache
uv cache clean

# Re-sync dependencies
uv sync --reinstall
```

### Python Version Mismatch

**Symptoms:**
```
Python 3.12+ required
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.12+
# Windows: Download from python.org
# Linux: sudo apt install python3.12
# macOS: brew install python@3.12
```

---

## Runtime Errors

### Model Not Found

**Symptoms:**
```
ERROR: Model 'deepseek-r1:7b' not found
```

**Solution:**
```bash
# Check Ollama is running
ollama list

# Pull the model
ollama pull deepseek-r1:7b

# Verify model is available
ollama list | grep deepseek
```

### OpenAI API Key Invalid

**Symptoms:**
```
ERROR: OpenAI client not initialized
```

**Solution:**
1. Check `.env` file exists:
   ```bash
   ls -la .env
   ```

2. Verify API key format (no quotes):
   ```bash
   OPENAI_API_KEY=sk-proj-...
   ```

3. Test API key:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

### ChromaDB Connection Error

**Symptoms:**
```
ERROR: Could not connect to ChromaDB
```

**Solution:**
```bash
# Check permissions
chmod -R 755 ./chroma_db

# If corrupted, reset database
rm -rf ./chroma_db

# Restart application
```

---

## Ingestion Issues

### PDF Loading Fails

**Symptoms:**
```
ERROR: Failed to load PDF
```

**Solution:**

**For Scanned PDFs:**
```bash
# Install Tesseract OCR
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract

# Verify installation
tesseract --version
```

**For Encrypted PDFs:**
```bash
# Install msoffcrypto-tool (already in dependencies)
uv sync

# Or manually decrypt
qpdf --decrypt input.pdf output.pdf
```

### Web Crawling Timeout

**Symptoms:**
```
ERROR: Request timeout after 120s
```

**Solution:**
1. Reduce crawl depth (try 0 or 1)
2. Reduce max pages (try 10)
3. Check internet connection
4. Try a different URL

### YouTube Transcript Unavailable

**Symptoms:**
```
ERROR: No transcript available
```

**Solution:**
1. **Check if video has captions:**
   - Open video on YouTube
   - Click "CC" button
   - If no captions, transcript unavailable

2. **Try different transcript language:**
   - Some videos only have non-English transcripts

3. **Use yt-dlp fallback:**
   - Already implemented in code
   - Should work automatically

---

## RAG Issues

### No Results Retrieved

**Symptoms:**
```
No relevant documents found
```

**Solution:**
1. **Lower similarity threshold:**
   ```python
   similarity_threshold = 0.1  # More permissive
   ```

2. **Increase Top-K:**
   ```python
   top_k = 10  # Retrieve more chunks
   ```

3. **Check documents are ingested:**
   - Go to **Metadata** tab
   - Verify documents are listed

4. **Try different RAG technique:**
   - Switch from Naive RAG to HyDE or RAG Fusion

### Poor Answer Quality

**Symptoms:**
- Irrelevant answers
- Hallucinations
- Missing information

**Solution:**
1. **Adjust chunking:**
   ```python
   chunk_size = 1024  # Larger chunks for more context
   chunk_overlap = 100  # More overlap
   ```

2. **Enable reranker:**
   ```python
   use_reranker = True
   ```

3. **Use better RAG technique:**
   - Try RAG Fusion or Hybrid Search

4. **Improve system prompt:**
   ```
   You are a helpful assistant. Answer based ONLY on the provided context.
   If the answer is not in the context, say "I don't have enough information."
   ```

### Slow Retrieval

**Symptoms:**
- Queries take >10 seconds
- UI freezes

**Solution:**
1. **Reduce Top-K:**
   ```python
   top_k = 3  # Faster retrieval
   ```

2. **Use faster embedding model:**
   ```python
   embedding_model = "all-minilm"  # Smaller, faster
   ```

3. **Enable caching:**
   - Already implemented for graphs
   - Embeddings cached automatically

---

## UI Issues

### Gradio Won't Start

**Symptoms:**
```
ERROR: Address already in use
```

**Solution:**
```bash
# Kill process on port 7860
# Windows:
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Linux/macOS:
lsof -ti:7860 | xargs kill -9

# Or use different port
uv run gradio src/ai_workdesk/ui/gradio_app.py --server-port 7861
```

### Authentication Fails

**Symptoms:**
- Login page doesn't appear
- Invalid credentials

**Solution:**
1. **Check credentials in `gradio_app.py`:**
   ```python
   auth=[("admin", "admin123"), ("demo", "demo123")]
   ```

2. **Clear browser cache:**
   - Ctrl+Shift+Delete
   - Clear cookies and cache

3. **Try incognito mode**

### Sidebar Toggle Not Working

**Symptoms:**
- Toggle button (☰) doesn't collapse sidebar

**Solution:**
1. **Hard refresh browser:**
   - Ctrl+Shift+R (Windows/Linux)
   - Cmd+Shift+R (macOS)

2. **Clear localStorage:**
   ```javascript
   // In browser console (F12)
   localStorage.clear()
   location.reload()
   ```

3. **Check browser console for errors:**
   - Press F12
   - Look for JavaScript errors

---

## Performance Issues

### High Memory Usage

**Symptoms:**
- Application uses >4GB RAM
- System becomes slow

**Solution:**
1. **Reduce batch size:**
   ```python
   BATCH_SIZE = 50  # In vector_store.py
   ```

2. **Clear cache:**
   ```bash
   rm -rf ./chroma_db/.cache
   ```

3. **Use smaller embedding model:**
   ```python
   embedding_model = "all-minilm"  # 384 dimensions vs 768
   ```

### Slow Graph Generation

**Symptoms:**
- Graph takes >30 seconds to generate

**Solution:**
1. **Reduce max nodes:**
   ```python
   max_nodes = 50  # Instead of 100
   ```

2. **Increase min weight:**
   ```python
   min_weight = 3  # Filter weak connections
   ```

3. **Use caching:**
   - Already implemented
   - Graph cached automatically

---

## Data Issues

### Documents Not Appearing in Metadata

**Symptoms:**
- Ingestion succeeds but metadata tab is empty

**Solution:**
1. **Check database file:**
   ```bash
   ls -la metadata.db
   ```

2. **Reset metadata database:**
   ```bash
   rm metadata.db
   # Restart application
   ```

3. **Re-ingest documents**

### Duplicate Documents

**Symptoms:**
- Same document appears multiple times

**Solution:**
1. **Check collection name:**
   - Different collections = different documents

2. **Clear collection:**
   - Use "Delete Collection" button
   - Re-ingest documents

---

## Obsidian Integration Issues

### Export Fails

**Symptoms:**
```
ERROR: Could not export to Obsidian
```

**Solution:**
1. **Check vault path in `.env`:**
   ```bash
   OBSIDIAN_VAULT_PATH=C:\path\to\vault
   ```

2. **Verify path exists:**
   ```bash
   ls -la "$OBSIDIAN_VAULT_PATH"
   ```

3. **Check write permissions:**
   ```bash
   # Windows: Right-click folder → Properties → Security
   # Linux/macOS: chmod -R 755 /path/to/vault
   ```

### Notes Not Syncing

**Symptoms:**
- Notes exported but not visible in Obsidian

**Solution:**
1. **Refresh Obsidian:**
   - Ctrl+R (Windows/Linux)
   - Cmd+R (macOS)

2. **Check subfolder:**
   - Notes are in `AI Workdesk/` subfolder
   - Navigate to: Vault → AI Workdesk

3. **Rebuild vault index:**
   - Obsidian Settings → Files & Links → Rebuild vault

---

## Logging and Debugging

### Enable Debug Logging

```bash
# In .env
LOG_LEVEL=DEBUG

# Restart application
```

### View Logs

```bash
# Tail logs in real-time
tail -f logs/ai_workdesk.log

# Windows PowerShell
Get-Content logs/ai_workdesk.log -Wait -Tail 50

# Search for errors
grep ERROR logs/ai_workdesk.log
```

### Common Log Messages

**INFO: Ingested X documents**
- Normal, documents successfully processed

**WARNING: Similarity threshold too high**
- Lower threshold for better retrieval

**ERROR: Model not found**
- Pull missing Ollama model

**ERROR: API key invalid**
- Check `.env` file

---

## Getting Help

### Before Asking for Help

1. **Check logs:**
   ```bash
   tail -50 logs/ai_workdesk.log
   ```

2. **Verify configuration:**
   ```bash
   cat .env
   ```

3. **Test basic functionality:**
   - Can you ingest a simple text file?
   - Does Naive RAG work?

### Reporting Issues

Include the following information:

1. **Environment:**
   - OS and version
   - Python version
   - UV version

2. **Error message:**
   - Full error traceback
   - Relevant log entries

3. **Steps to reproduce:**
   - What were you doing?
   - What did you expect?
   - What actually happened?

4. **Configuration:**
   - Relevant `.env` settings (hide API keys!)
   - Model being used
   - RAG technique

### Community Resources

- **GitHub Issues:** [github.com/rderakhshan/Work-Desk/issues](https://github.com/rderakhshan/Work-Desk/issues)
- **Documentation:** Check other sections of this guide
- **Logs:** Always check `logs/ai_workdesk.log` first

---

## Next Steps

- [Configuration](configuration.md) - Detailed configuration guide
- [Advanced Topics](advanced.md) - RAG techniques and optimization
- [API Reference](api-reference.md) - Developer documentation

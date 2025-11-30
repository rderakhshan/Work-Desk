# API Reference

Developer documentation for AI Workdesk core classes and methods.

## Core Classes

### VectorStoreManager

Manages vector database operations for document storage and retrieval.

**Location:** `src/ai_workdesk/rag/vector_store.py`

#### Initialization

```python
from ai_workdesk.rag.vector_store import VectorStoreManager

vector_store = VectorStoreManager(
    persist_directory="./chroma_db",
    collection_name="my_documents",
    embedding_provider="ollama",
    embedding_model="nomic-embed-text"
)
```

#### Methods

**`add_documents(documents: List[Document]) -> None`**

Add documents to the vector store.

```python
from langchain.schema import Document

docs = [
    Document(page_content="Text content", metadata={"source": "file.txt"}),
    Document(page_content="More content", metadata={"source": "file2.txt"})
]

vector_store.add_documents(docs)
```

**`similarity_search(query: str, k: int = 4, threshold: float = 0.1) -> List[Document]`**

Retrieve similar documents.

```python
results = vector_store.similarity_search(
    query="What is RAG?",
    k=5,
    threshold=0.2
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

**`delete_collection() -> None`**

Delete the current collection.

```python
vector_store.delete_collection()
```

**`get_all_embeddings() -> Dict`**

Get all embeddings and metadata.

```python
data = vector_store.get_all_embeddings()
embeddings = data["embeddings"]  # List of vectors
metadatas = data["metadatas"]    # List of metadata dicts
```

---

### DocumentProcessor

Processes and chunks documents for ingestion.

**Location:** `src/ai_workdesk/rag/ingestion.py`

#### Initialization

```python
from ai_workdesk.rag.ingestion import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=512,
    chunk_overlap=50,
    separator="\n\n"
)
```

#### Methods

**`process_file(file_path: str) -> List[Document]`**

Process a single file.

```python
documents = processor.process_file("document.pdf")
```

**`process_directory(directory: str) -> List[Document]`**

Process all files in a directory.

```python
documents = processor.process_directory("./docs")
```

**`chunk_documents(documents: List[Document]) -> List[Document]`**

Chunk documents into smaller pieces.

```python
chunked_docs = processor.chunk_documents(documents)
```

---

### MetadataStore

Tracks ingested document metadata.

**Location:** `src/ai_workdesk/rag/metadata_store.py`

#### Initialization

```python
from ai_workdesk.rag.metadata_store import MetadataStore

metadata_store = MetadataStore(db_path="metadata.db")
```

#### Methods

**`add_document(filename: str, file_size: int, doc_type: str) -> int`**

Add document metadata.

```python
doc_id = metadata_store.add_document(
    filename="report.pdf",
    file_size=1024000,
    doc_type="pdf"
)
```

**`get_all_documents(limit: int = 100, offset: int = 0) -> List[Dict]`**

Retrieve all document metadata.

```python
documents = metadata_store.get_all_documents(limit=50, offset=0)

for doc in documents:
    print(f"{doc['filename']} - {doc['upload_time']}")
```

**`delete_document(doc_id: int) -> None`**

Delete document metadata.

```python
metadata_store.delete_document(doc_id=1)
```

---

### GraphRAG

Knowledge graph construction and retrieval.

**Location:** `src/ai_workdesk/rag/graph_rag.py`

#### Initialization

```python
from ai_workdesk.rag.graph_rag import GraphRAG

graph_rag = GraphRAG(
    vector_store=vector_store,
    entity_types=["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]
)
```

#### Methods

**`build_graph(documents: List[Document]) -> nx.Graph`**

Build knowledge graph from documents.

```python
import networkx as nx

graph = graph_rag.build_graph(documents)

# Graph statistics
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

**`visualize_graph(graph: nx.Graph, output_path: str) -> str`**

Generate interactive graph visualization.

```python
html_path = graph_rag.visualize_graph(
    graph=graph,
    output_path="graph.html"
)
```

**`graph_retrieval(query: str, graph: nx.Graph, k: int = 5) -> List[Document]`**

Retrieve documents using graph traversal.

```python
results = graph_rag.graph_retrieval(
    query="Who founded OpenAI?",
    graph=graph,
    k=5
)
```

---

### EmbeddingVisualizer

Visualize embeddings in 2D/3D.

**Location:** `src/ai_workdesk/rag/visualization.py`

#### Initialization

```python
from ai_workdesk.rag.visualization import EmbeddingVisualizer

visualizer = EmbeddingVisualizer()
```

#### Methods

**`project_embeddings_2d(embeddings: np.ndarray, method: str = "umap") -> Tuple[np.ndarray, go.Figure]`**

Project embeddings to 2D.

```python
import numpy as np

embeddings = np.random.rand(100, 768)  # Example embeddings

projected, fig = visualizer.project_embeddings_2d(
    embeddings=embeddings,
    labels=["Doc 1", "Doc 2", ...],
    method="umap"  # or "tsne", "pca"
)

fig.show()  # Display Plotly figure
```

**`project_embeddings_3d(embeddings: np.ndarray, method: str = "umap") -> Tuple[np.ndarray, go.Figure]`**

Project embeddings to 3D.

```python
projected, fig = visualizer.project_embeddings_3d(
    embeddings=embeddings,
    labels=["Doc 1", "Doc 2", ...],
    method="umap"
)

fig.show()
```

---

### ObsidianExporter

Export notes to Obsidian vault.

**Location:** `src/ai_workdesk/tools/obsidian_exporter.py`

#### Initialization

```python
from ai_workdesk.tools.obsidian_exporter import ObsidianExporter

exporter = ObsidianExporter(
    vault_path="C:/path/to/vault"
)
```

#### Methods

**`export_youtube_note(video_id: str, title: str, summary: str, transcript: str) -> str`**

Export YouTube video note.

```python
note_path = exporter.export_youtube_note(
    video_id="dQw4w9WgXcQ",
    title="Video Title",
    summary="Video summary...",
    transcript="Full transcript..."
)
```

**`export_chat_note(chat_history: List[Dict], title: str) -> str`**

Export chat conversation.

```python
chat_history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

note_path = exporter.export_chat_note(
    chat_history=chat_history,
    title="Chat Session"
)
```

**`export_rag_note(query: str, answer: str, sources: List[str]) -> str`**

Export RAG session.

```python
note_path = exporter.export_rag_note(
    query="What is RAG?",
    answer="RAG stands for...",
    sources=["doc1.pdf", "doc2.txt"]
)
```

---

### OllamaClient

Wrapper for Ollama LLM interactions.

**Location:** `src/ai_workdesk/tools/llm/ollama_client.py`

#### Initialization

```python
from ai_workdesk.tools.llm.ollama_client import OllamaClient

client = OllamaClient(
    model="deepseek-r1:7b",
    base_url="http://localhost:11434",
    temperature=0.7,
    max_tokens=2000
)
```

#### Methods

**`chat(messages: List[Dict], model: str = None) -> str`**

Generate chat completion.

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

response = client.chat(messages)
print(response)
```

**`list_models() -> List[str]`**

List available Ollama models.

```python
models = client.list_models()
print(models)  # ['deepseek-r1:7b', 'llama3.2', ...]
```

**`embed(text: str) -> List[float]`**

Generate text embedding.

```python
embedding = client.embed("Sample text")
print(len(embedding))  # 768 (for nomic-embed-text)
```

---

## Configuration Classes

### Settings

Application configuration.

**Location:** `src/ai_workdesk/core/config.py`

```python
from ai_workdesk.core.config import Settings

settings = Settings()

# Access configuration
print(settings.default_llm_model)
print(settings.ollama_base_url)
print(settings.log_level)
```

**Attributes:**
- `environment: str` - Application environment
- `log_level: str` - Logging level
- `log_file: Path` - Log file path
- `default_llm_model: str` - Default LLM model
- `default_embedding_provider: str` - Default embedding provider
- `default_temperature: float` - Default temperature
- `max_tokens: int` - Maximum tokens
- `ollama_base_url: str` - Ollama server URL
- `ollama_chat_model: str` - Ollama chat model
- `ollama_embedding_model: str` - Ollama embedding model
- `openai_api_key: str` - OpenAI API key

---

## Utility Functions

### Document Loading

**`load_pdf(file_path: str) -> List[Document]`**

```python
from ai_workdesk.rag.ingestion import load_pdf

documents = load_pdf("document.pdf")
```

**`load_text(file_path: str) -> List[Document]`**

```python
from ai_workdesk.rag.ingestion import load_text

documents = load_text("document.txt")
```

### Text Processing

**`sanitize_filename(filename: str) -> str`**

```python
from ai_workdesk.tools.obsidian_exporter import sanitize_filename

safe_name = sanitize_filename("My File: Test.md")
# Returns: "My File Test.md"
```

### Embedding Utilities

**`get_embedding_function(provider: str, model: str) -> Embeddings`**

```python
from ai_workdesk.rag.vector_store import get_embedding_function

embeddings = get_embedding_function(
    provider="ollama",
    model="nomic-embed-text"
)
```

---

## Data Models

### Document

LangChain document schema.

```python
from langchain.schema import Document

doc = Document(
    page_content="Text content here",
    metadata={
        "source": "file.txt",
        "page": 1,
        "chunk_id": 0
    }
)
```

### DashboardCard

Smart dashboard item.

```python
from ai_workdesk.smart_dashboard.models import DashboardCard, SourceType

card = DashboardCard(
    title="News Article",
    summary="Article summary...",
    source_type=SourceType.NEWS,
    source_link="https://example.com",
    urgency_score=0.8,
    timestamp=datetime.now(),
    metadata={"author": "John Doe"}
)
```

---

## Example Workflows

### Complete RAG Pipeline

```python
from ai_workdesk.rag.ingestion import DocumentProcessor
from ai_workdesk.rag.vector_store import VectorStoreManager
from ai_workdesk.tools.llm.ollama_client import OllamaClient

# 1. Process documents
processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
documents = processor.process_file("document.pdf")

# 2. Store in vector database
vector_store = VectorStoreManager(
    collection_name="my_docs",
    embedding_provider="ollama"
)
vector_store.add_documents(documents)

# 3. Retrieve relevant chunks
query = "What is the main topic?"
results = vector_store.similarity_search(query, k=4)

# 4. Generate answer
client = OllamaClient(model="deepseek-r1:7b")
context = "\n\n".join([doc.page_content for doc in results])
messages = [
    {"role": "system", "content": "Answer based on the context."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
]
answer = client.chat(messages)
print(answer)
```

### Knowledge Graph Construction

```python
from ai_workdesk.rag.graph_rag import GraphRAG
from ai_workdesk.rag.ingestion import DocumentProcessor

# 1. Process documents
processor = DocumentProcessor()
documents = processor.process_directory("./docs")

# 2. Build graph
graph_rag = GraphRAG(vector_store=vector_store)
graph = graph_rag.build_graph(documents)

# 3. Visualize
html_path = graph_rag.visualize_graph(graph, "knowledge_graph.html")

# 4. Query graph
results = graph_rag.graph_retrieval("Who are the key people?", graph, k=5)
```

---

## Next Steps

- [Configuration](configuration.md) - Detailed configuration guide
- [Advanced Topics](advanced.md) - RAG techniques and optimization
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

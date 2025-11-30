# Advanced Topics

Deep dive into RAG techniques, optimization strategies, and advanced features.

## RAG Techniques Explained

### Naive RAG

**Description:** Direct similarity search using vector embeddings.

**How It Works:**
1. User query is embedded using the same model as documents
2. Vector similarity search finds top-K most similar chunks
3. Retrieved chunks are passed to LLM as context
4. LLM generates answer based on context

**Pros:**
- Fastest retrieval method
- Simple and reliable
- Low computational cost

**Cons:**
- May miss semantically related content
- No query understanding
- Limited to surface-level similarity

**Best For:**
- Quick lookups
- Well-defined queries
- Small to medium document collections

**Configuration:**
```python
rag_technique = "Naive RAG"
top_k = 4
similarity_threshold = 0.1
```

---

### HyDE (Hypothetical Document Embeddings)

**Description:** Generates a hypothetical answer first, then searches for similar documents.

**How It Works:**
1. LLM generates a hypothetical answer to the query
2. Hypothetical answer is embedded
3. Vector search finds documents similar to the hypothetical answer
4. Retrieved documents are used to generate the actual answer

**Pros:**
- Better semantic matching
- Handles complex queries well
- Finds conceptually related content

**Cons:**
- Slower (requires extra LLM call)
- Higher token usage
- May hallucinate in hypothetical answer

**Best For:**
- Complex, open-ended questions
- Conceptual queries
- Large document collections

**Configuration:**
```python
rag_technique = "HyDE"
top_k = 6  # Retrieve more for better coverage
temperature = 0.3  # Lower for hypothetical answer generation
```

---

### RAG Fusion

**Description:** Generates multiple query variations, retrieves for each, and merges results.

**How It Works:**
1. LLM generates 3-5 query variations
2. Each variation performs vector search
3. Results are merged using Reciprocal Rank Fusion (RRF)
4. Top-ranked chunks are used for final answer

**Pros:**
- Comprehensive coverage
- Handles ambiguous queries
- Reduces retrieval bias

**Cons:**
- Slowest method (multiple searches)
- High computational cost
- May retrieve redundant content

**Best For:**
- Ambiguous queries
- Multi-faceted questions
- Critical applications requiring thoroughness

**Configuration:**
```python
rag_technique = "RAG Fusion"
top_k = 8  # Higher for better fusion
num_queries = 4  # Number of query variations
```

**RRF Formula:**
```
RRF_score(doc) = Σ(1 / (k + rank_i))
where k = 60 (constant), rank_i = rank in query i
```

---

### Hybrid Search

**Description:** Combines vector similarity (semantic) with BM25 (keyword) search.

**How It Works:**
1. Vector search finds semantically similar chunks
2. BM25 search finds keyword matches
3. Results are combined using weighted scoring
4. Top-ranked chunks are returned

**Pros:**
- Balances semantic and lexical matching
- Handles both conceptual and specific queries
- More robust than pure vector search

**Cons:**
- Moderate computational cost
- Requires BM25 index maintenance
- Tuning weights can be tricky

**Best For:**
- Mixed query types
- Technical documentation
- Queries with specific terms

**Configuration:**
```python
rag_technique = "Hybrid Search"
top_k = 5
alpha = 0.5  # 0.5 = equal weight, 1.0 = pure vector, 0.0 = pure BM25
```

**Scoring Formula:**
```
hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
```

---

### Graph+Vector RAG

**Description:** Uses knowledge graph traversal combined with vector search.

**How It Works:**
1. Extract entities from query using NER
2. Find related entities in knowledge graph
3. Retrieve documents connected to entities
4. Combine with vector search results
5. Generate answer with entity-aware context

**Pros:**
- Entity-aware retrieval
- Captures relationships
- Excellent for factual queries

**Cons:**
- Requires graph construction
- Slower than pure vector search
- Graph quality affects results

**Best For:**
- Entity-centric queries
- Relationship questions
- Structured knowledge domains

**Configuration:**
```python
rag_technique = "Graph+Vector"
top_k = 6
graph_depth = 2  # How many hops in graph
min_entity_frequency = 2  # Filter rare entities
```

**Entity Types:**
- PERSON - People, authors, experts
- ORG - Organizations, companies
- GPE - Locations, countries, cities
- PRODUCT - Products, tools, technologies
- EVENT - Events, conferences, releases

---

## Performance Optimization

### Embedding Optimization

**Model Selection:**

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| nomic-embed-text | 768 | ⚡⚡⚡ | ⭐⭐⭐ | General purpose, fast |
| mxbai-embed-large | 1024 | ⚡⚡ | ⭐⭐⭐⭐ | High quality, slower |
| all-minilm | 384 | ⚡⚡⚡⚡ | ⭐⭐ | Speed-critical apps |
| text-embedding-3-large | 3072 | ⚡ | ⭐⭐⭐⭐⭐ | Best quality, cloud |

**Batch Processing:**
```python
# Process documents in batches
BATCH_SIZE = 100

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    embeddings = embed_batch(batch)
    vector_store.add(embeddings)
```

### Chunking Strategies

**Fixed-Size Chunking:**
```python
chunk_size = 512
chunk_overlap = 50
# Best for: General documents
```

**Semantic Chunking:**
```python
# Split on paragraphs, then merge to target size
separator = "\n\n"
# Best for: Well-structured documents
```

**Recursive Chunking:**
```python
# Try multiple separators in order
separators = ["\n\n", "\n", ". ", " "]
# Best for: Mixed content types
```

### Caching Strategies

**Graph Caching:**
```python
# Cache knowledge graph to avoid rebuilds
cache_key = hash(document_ids)
if cache_key in graph_cache:
    return graph_cache[cache_key]
```

**Embedding Caching:**
```python
# Cache embeddings for frequently accessed documents
embedding_cache = LRUCache(maxsize=1000)
```

### Query Optimization

**Query Expansion:**
```python
# Add synonyms and related terms
expanded_query = expand_with_synonyms(query)
```

**Query Rewriting:**
```python
# Simplify complex queries
rewritten = llm.rewrite_query(query, style="simple")
```

---

## Knowledge Graph Construction

### Entity Extraction

**Using spaCy:**
```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
```

**Entity Types:**
- **PERSON:** Barack Obama, John Smith
- **ORG:** Google, Microsoft, OpenAI
- **GPE:** United States, London, California
- **PRODUCT:** iPhone, Windows, ChatGPT
- **EVENT:** World War II, Olympics 2024

### Relationship Extraction

**Co-occurrence Method:**
```python
# Entities appearing in same chunk are related
if entity1 in chunk and entity2 in chunk:
    graph.add_edge(entity1, entity2, weight=1)
```

**Distance-based Weighting:**
```python
# Closer entities have stronger relationships
distance = abs(pos1 - pos2)
weight = 1 / (1 + distance)
```

### Graph Visualization

**Node Sizing:**
```python
# Size by degree (number of connections)
node_size = degree * 10 + 20
```

**Community Detection:**
```python
# Group related entities
communities = detect_communities(graph)
node_color = community_id
```

---

## Multimodal RAG

### Image Processing

**OCR for Scanned Documents:**
```python
from PIL import Image
import pytesseract

image = Image.open("document.png")
text = pytesseract.image_to_string(image)
```

**Image Embeddings (Future):**
```python
# Using CLIP or similar
image_embedding = clip_model.encode_image(image)
```

### Audio Processing

**Transcription:**
```python
from faster_whisper import WhisperModel

model = WhisperModel("base")
segments, info = model.transcribe("audio.mp3")
text = " ".join([seg.text for seg in segments])
```

---

## Evaluation Metrics

### Retrieval Quality

**Precision@K:**
```python
precision = relevant_retrieved / k
```

**Recall@K:**
```python
recall = relevant_retrieved / total_relevant
```

**MRR (Mean Reciprocal Rank):**
```python
mrr = 1 / rank_of_first_relevant
```

### Answer Quality

**BLEU Score:**
```python
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu([reference], candidate)
```

**ROUGE Score:**
```python
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(candidate, reference)
```

**Semantic Similarity:**
```python
similarity = cosine_similarity(
    embed(generated_answer),
    embed(ground_truth)
)
```

---

## Production Deployment

### Scaling Strategies

**Horizontal Scaling:**
- Deploy multiple Gradio instances
- Use load balancer (nginx, HAProxy)
- Share ChromaDB via network storage

**Vertical Scaling:**
- Increase RAM for larger vector stores
- Use GPU for faster embeddings
- SSD for faster disk I/O

### Monitoring

**Key Metrics:**
- Query latency (p50, p95, p99)
- Retrieval accuracy
- LLM token usage
- Error rates

**Logging:**
```python
logger.info(f"Query: {query}")
logger.info(f"Retrieved: {len(chunks)} chunks")
logger.info(f"Latency: {latency_ms}ms")
```

### Security

**API Key Rotation:**
```bash
# Rotate keys monthly
OPENAI_API_KEY=new-key
```

**Rate Limiting:**
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@limiter.limit("10/minute")
def query_endpoint():
    ...
```

---

## Next Steps

- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [API Reference](api-reference.md) - Developer documentation
- [Configuration](configuration.md) - Detailed configuration guide

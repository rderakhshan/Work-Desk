"""
UI Constants for AI Workdesk.
"""

# Constants for selections
EMBEDDING_MODELS = ["OpenAI", "HuggingFace", "Ollama", "Google Gemini"]
DATABASES = ["ChromaDB", "FAISS", "PostgreSQL (PGVector)", "SQLite", "Pinecone"]

# Provider-based model lists
MODELS = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    "Ollama": ["deepseek-r1:7b", "gemma3:4b", "llama3", "mistral", "phi3"]
}

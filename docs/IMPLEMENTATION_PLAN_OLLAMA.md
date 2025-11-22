# Implementation Plan: Ollama Integration

*(See the detailed plan in .gemini/antigravity/brain/.../implementation_plan.md)*

- Add `langchain-ollama` dependency
- Update `.env.example` with Ollama settings
- Extend `Settings` to include Ollama config
- Refactor `VectorStore` to support multiple embedding providers (HuggingFace, OpenAI, Ollama)
- Update Gradio UI to use Ollama models as default and allow dynamic model discovery
- Add helper `OllamaClient` for chat and embeddings
- Create manual test script for Ollama integration
- Verify end‑to‑end RAG flow with local models
- Set Ollama (`gemma3:4b`) as default chat model and `nomic-embed-text` as default embedding model

All changes are backward compatible.

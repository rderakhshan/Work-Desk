from langchain_ollama import ChatOllama
from ai_workdesk.core.config import get_settings

class OllamaClient:
    """Simple wrapper around LangChain's ChatOllama for the AI Workdesk.

    Provides a `chat` method that returns the response text given a user message.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.7, max_tokens: int = 4096):
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = model or settings.ollama_chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ChatOllama(
            model=self.model, 
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens  # Ollama uses num_predict for max tokens
        )

    def chat(self, message: str | list[dict], model: str | None = None) -> str:
        """Send a message to the Ollama model and return the response.

        Args:
            message: The user prompt string OR a list of message dicts [{"role": "user", "content": "..."}]
            model: Optional model override for this specific call.
        Returns:
            The assistant's response as a string.
        """
        # If a specific model is requested, we might need to re-instantiate or use a different approach
        # Since LangChain's ChatOllama is bound to a model, we can try to pass it in invoke if supported,
        # or just create a temporary client if the model differs.
        
        target_model = model or self.model
        
        # If the model is different from the initialized one, create a temp client
        if target_model != self.model:
            temp_client = ChatOllama(
                model=target_model, 
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
            client_to_use = temp_client
        else:
            client_to_use = self.client

        # Handle input format
        if isinstance(message, str):
            messages = [{"role": "user", "content": message}]
        else:
            messages = message

        response = client_to_use.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    def list_models(self) -> list[str]:
        """List available models from the local Ollama instance."""
        import httpx
        try:
            response = httpx.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return ["llama3", "mistral"] # Fallback
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return ["llama3", "mistral"] # Fallback

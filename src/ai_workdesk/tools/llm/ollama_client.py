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

    def chat(self, message: str) -> str:
        """Send a single user message to the Ollama model and return the response.

        Args:
            message: The user prompt.
        Returns:
            The assistant's response as a string.
        """
        response = self.client.invoke([{"role": "user", "content": message}])
        return response.content if hasattr(response, 'content') else str(response)

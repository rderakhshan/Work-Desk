from langchain_ollama import ChatOllama
from ai_workdesk.core.config import get_settings

class OllamaClient:
    """Simple wrapper around LangChain's ChatOllama for the AI Workdesk.

    Provides a `chat` method that returns the response text given a user message.
    """

    def __init__(self, model: str | None = None):
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = model or settings.ollama_chat_model
        self.client = ChatOllama(model=self.model, base_url=self.base_url)

    def chat(self, message: str) -> str:
        """Send a single user message to the Ollama model and return the response.

        Args:
            message: The user prompt.
        Returns:
            The assistant's response as a string.
        """
        response = self.client.invoke([{"role": "user", "content": message}])
        return response.get("content", "")

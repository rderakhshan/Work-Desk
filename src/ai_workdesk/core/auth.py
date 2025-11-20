"""
Authentication and API key management for AI Workdesk.

This module provides secure handling of API keys and credentials
for various AI services while following best practices.
"""

import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings

console = Console()


class AuthenticationManager:
    """Manages authentication for various AI services."""

    def __init__(self) -> None:
        """Initialize the authentication manager."""
        self.settings = get_settings()

    def get_openai_api_key(self) -> str | None:
        """
        Get OpenAI API key from settings or environment.

        Returns:
            API key if available, None otherwise
        """
        return self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")

    def get_anthropic_api_key(self) -> str | None:
        """
        Get Anthropic API key from settings or environment.

        Returns:
            API key if available, None otherwise
        """
        return self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    def get_google_api_key(self) -> str | None:
        """
        Get Google AI API key from settings or environment.

        Returns:
            API key if available, None otherwise
        """
        return self.settings.google_api_key or os.getenv("GOOGLE_API_KEY")

    def get_cohere_api_key(self) -> str | None:
        """
        Get Cohere API key from settings or environment.

        Returns:
            API key if available, None otherwise
        """
        return self.settings.cohere_api_key or os.getenv("COHERE_API_KEY")

    def check_authentication_status(self) -> dict[str, bool]:
        """
        Check which API keys are configured.

        Returns:
            Dictionary mapping service names to authentication status
        """
        return {
            "OpenAI": self.get_openai_api_key() is not None,
            "Anthropic": self.get_anthropic_api_key() is not None,
            "Google AI": self.get_google_api_key() is not None,
            "Cohere": self.get_cohere_api_key() is not None,
        }

    def display_authentication_status(self) -> None:
        """Display authentication status in a formatted table."""
        status = self.check_authentication_status()

        table = Table(title="🔐 Authentication Status", show_header=True, header_style="bold cyan")
        table.add_column("Service", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("API Key", style="dim")

        for service, authenticated in status.items():
            if authenticated:
                status_emoji = "✅"
                status_text = "Authenticated"
                key_preview = self._get_key_preview(service)
            else:
                status_emoji = "❌"
                status_text = "Not Configured"
                key_preview = "N/A"

            table.add_row(service, f"{status_emoji} {status_text}", key_preview)

        console.print(table)

        # Show warning if no keys are configured
        if not any(status.values()):
            console.print(
                Panel(
                    "[yellow]⚠️  No API keys configured! "
                    "Please set your API keys in .env file.[/yellow]\n\n"
                    "Example: Copy .env.example to .env and add your keys.",
                    title="⚠️  Warning",
                    border_style="yellow",
                )
            )

    def _get_key_preview(self, service: str) -> str:
        """
        Get a preview of the API key (first 8 and last 4 characters).

        Args:
            service: Service name

        Returns:
            Masked API key preview
        """
        key_getters = {
            "OpenAI": self.get_openai_api_key,
            "Anthropic": self.get_anthropic_api_key,
            "Google AI": self.get_google_api_key,
            "Cohere": self.get_cohere_api_key,
        }

        getter = key_getters.get(service)
        if not getter:
            return "N/A"

        api_key = getter()
        if not api_key:
            return "N/A"

        if len(api_key) <= 12:
            return "*" * len(api_key)

        return f"{api_key[:8]}...{api_key[-4:]}"

    def validate_service(self, service: str) -> bool:
        """
        Validate if a specific service is authenticated.

        Args:
            service: Service name (openai, anthropic, google, cohere)

        Returns:
            True if authenticated, False otherwise

        Raises:
            ValueError: If service name is not recognized
        """
        service_lower = service.lower()
        validators = {
            "openai": self.get_openai_api_key,
            "anthropic": self.get_anthropic_api_key,
            "google": self.get_google_api_key,
            "cohere": self.get_cohere_api_key,
        }

        if service_lower not in validators:
            raise ValueError(
                f"Unknown service: {service}. Available services: {', '.join(validators.keys())}"
            )

        return validators[service_lower]() is not None


# Global authentication manager instance
_auth_manager: AuthenticationManager | None = None


def get_auth_manager() -> AuthenticationManager:
    """
    Get the global authentication manager instance.

    Returns:
        AuthenticationManager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager

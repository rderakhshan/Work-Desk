"""
Example: Using Multiple AI Services

This example shows how to work with multiple AI providers
(OpenAI, Anthropic) using the AI Workdesk authentication system.

Run with: uv run python examples/multi_service_example.py
"""

from ai_workdesk import get_settings, get_auth_manager, get_logger
from rich.console import Console
from rich.panel import Panel

console = Console()
logger = get_logger(__name__)


def chat_with_openai(message: str) -> str | None:
    """Chat with OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        console.print("[yellow]OpenAI not installed. Run: uv sync --extra llm[/yellow]")
        return None

    settings = get_settings()
    auth = get_auth_manager()

    if not auth.validate_service("openai"):
        console.print("[yellow]OpenAI not configured[/yellow]")
        return None

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.default_llm_model,
        messages=[{"role": "user", "content": message}],
        temperature=settings.default_temperature,
    )

    return response.choices[0].message.content


def chat_with_anthropic(message: str) -> str | None:
    """Chat with Anthropic Claude."""
    try:
        from anthropic import Anthropic
    except ImportError:
        console.print("[yellow]Anthropic not installed. Run: uv sync --extra llm[/yellow]")
        return None

    settings = get_settings()
    auth = get_auth_manager()

    if not auth.validate_service("anthropic"):
        console.print("[yellow]Anthropic not configured[/yellow]")
        return None

    client = Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=settings.max_tokens,
        messages=[{"role": "user", "content": message}],
    )

    return response.content[0].text


def main() -> None:
    """Main function demonstrating multi-service usage."""
    console.print("\n[bold cyan]🤖 Multi-Service AI Example[/bold cyan]\n")

    # Show authentication status
    auth = get_auth_manager()
    auth.display_authentication_status()

    test_message = "What is Python in one sentence?"

    console.print(f"\n[cyan]Test Message:[/cyan] {test_message}\n")

    # Try OpenAI
    console.print("[bold]OpenAI Response:[/bold]")
    openai_response = chat_with_openai(test_message)
    if openai_response:
        console.print(f"[green]{openai_response}[/green]\n")
    else:
        console.print("[dim]Not configured or not installed[/dim]\n")

    # Try Anthropic
    console.print("[bold]Anthropic Response:[/bold]")
    anthropic_response = chat_with_anthropic(test_message)
    if anthropic_response:
        console.print(f"[green]{anthropic_response}[/green]\n")
    else:
        console.print("[dim]Not configured or not installed[/dim]\n")


if __name__ == "__main__":
    main()

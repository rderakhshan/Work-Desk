"""
Example: Simple OpenAI Chat Interface

This example demonstrates how to use the configured OpenAI API
with the AI Workdesk configuration management.

Run with: uv run python examples/simple_openai_chat.py
"""

from ai_workdesk import get_settings, get_logger
from rich.console import Console
from rich.panel import Panel

console = Console()
logger = get_logger(__name__)


def test_openai_connection() -> None:
    """Test OpenAI API connection with a simple completion."""
    try:
        # Import OpenAI (requires: uv sync --extra llm)
        from openai import OpenAI
    except ImportError:
        console.print(
            Panel(
                "[yellow]⚠️  OpenAI package not installed![/yellow]\n\n"
                "Install LLM tools with:\n"
                "[cyan]uv sync --extra llm[/cyan]",
                title="Missing Dependency",
                border_style="yellow",
            )
        )
        return

    # Get settings
    settings = get_settings()

    console.print("\n[bold cyan]🤖 OpenAI Chat Test[/bold cyan]\n")
    console.print(f"[dim]Model: {settings.default_llm_model}[/dim]")
    console.print(f"[dim]Temperature: {settings.default_temperature}[/dim]\n")

    # Initialize OpenAI client
    client = OpenAI(api_key=settings.openai_api_key)

    # Test message
    test_message = "Hello! Can you explain what AI is in one sentence?"

    console.print(f"[cyan]User:[/cyan] {test_message}\n")
    logger.info(f"Sending request to {settings.default_llm_model}")

    try:
        # Create completion
        response = client.chat.completions.create(
            model=settings.default_llm_model,
            messages=[{"role": "user", "content": test_message}],
            temperature=settings.default_temperature,
            max_tokens=settings.max_tokens,
        )

        # Extract response
        ai_response = response.choices[0].message.content

        console.print(f"[green]AI:[/green] {ai_response}\n")
        logger.info("Response received successfully")

        # Show token usage
        usage = response.usage
        console.print("[dim]Token Usage:[/dim]")
        console.print(f"[dim]  Prompt: {usage.prompt_tokens}[/dim]")
        console.print(f"[dim]  Completion: {usage.completion_tokens}[/dim]")
        console.print(f"[dim]  Total: {usage.total_tokens}[/dim]\n")

        console.print("[green]✅ OpenAI connection successful![/green]\n")

    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]\n")
        logger.error(f"OpenAI API error: {e}")


if __name__ == "__main__":
    test_openai_connection()

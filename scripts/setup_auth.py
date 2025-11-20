"""
Authentication setup and verification script.

Run this script to:
1. Check if .env file exists
2. Verify API keys are configured
3. Test authentication manager
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from ai_workdesk import get_auth_manager, get_settings

console = Console()


def main() -> None:
    """Main function."""
    console.print("\n[bold cyan]🔐 AI Workdesk Authentication Setup[/bold cyan]\n")

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        console.print(
            Panel(
                "[yellow]⚠️  .env file not found![/yellow]\n\n"
                "Creating .env file from .env.example...",
                title="Setup Required",
                border_style="yellow",
            )
        )

        # Copy .env.example to .env
        env_example = Path(".env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            console.print("[green]✅ Created .env file[/green]")
            console.print(
                "\n[yellow]📝 Please edit .env and add your API keys, then run this script again.[/yellow]\n"
            )
            return
        else:
            console.print("[red]❌ .env.example not found![/red]")
            return

    # Load settings and check authentication
    console.print("[cyan]Loading configuration...[/cyan]")
    settings = get_settings()

    console.print(f"\n[dim]Environment: {settings.environment}[/dim]")
    console.print(f"[dim]Log Level: {settings.log_level}[/dim]\n")

    # Display authentication status
    auth_manager = get_auth_manager()
    auth_manager.display_authentication_status()

    # Check if any keys are configured
    status = auth_manager.check_authentication_status()
    if any(status.values()):
        console.print("\n[green]✅ Authentication setup complete![/green]")

        # Ask if user wants to test a service
        if Confirm.ask("\n[cyan]Would you like to test your API keys?[/cyan]"):
            test_apis(auth_manager)
    else:
        console.print(
            "\n[yellow]⚠️  No API keys configured. Please edit .env file and add your keys.[/yellow]"
        )


def test_apis(auth_manager) -> None:
    """Test configured API services."""
    console.print("\n[cyan]🧪 Testing API Services...[/cyan]\n")

    # Test OpenAI
    if auth_manager.validate_service("openai"):
        try:
            console.print("[cyan]Testing OpenAI...[/cyan]", end=" ")
            # Note: Actual API test would go here when openai is installed
            console.print("[green]✅ Key configured[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    # Test Anthropic
    if auth_manager.validate_service("anthropic"):
        try:
            console.print("[cyan]Testing Anthropic...[/cyan]", end=" ")
            # Note: Actual API test would go here when anthropic is installed
            console.print("[green]✅ Key configured[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    console.print(
        "\n[dim]Note: Full API testing requires installing optional dependencies:[/dim]"
    )
    console.print("[dim]  uv sync --extra llm[/dim]\n")


if __name__ == "__main__":
    main()

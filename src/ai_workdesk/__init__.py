"""AI Workdesk - A professional workdesk for developing and utilizing AI tools."""

__version__ = "0.1.0"
__author__ = "rderakhshan"

from .core.auth import AuthenticationManager, get_auth_manager
from .core.config import Settings, get_settings
from .core.logger import get_logger, setup_logger

__all__ = [
    "__version__",
    "__author__",
    # Config
    "Settings",
    "get_settings",
    # Auth
    "AuthenticationManager",
    "get_auth_manager",
    # Logger
    "get_logger",
    "setup_logger",
]


def main() -> None:
    """Main entry point for the CLI."""
    from rich.console import Console

    console = Console()

    console.print("\n[bold cyan]🚀 AI Workdesk[/bold cyan]", justify="center")
    console.print(
        "[dim]A professional workdesk for developing and utilizing AI tools[/dim]\n",
        justify="center",
    )

    # Display authentication status
    auth_manager = get_auth_manager()
    auth_manager.display_authentication_status()

    console.print(
        "\n[green]✨ AI Workdesk initialized successfully![/green]",
        justify="center",
    )
    console.print(
        "\n[dim]Import modules: from ai_workdesk import get_settings, get_auth_manager[/dim]",
        justify="center",
    )


if __name__ == "__main__":
    main()

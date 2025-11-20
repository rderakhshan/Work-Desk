"""
Logging configuration for AI Workdesk.

This module sets up structured logging using Loguru with
custom formatting and handlers.
"""

import sys
from pathlib import Path

from loguru import logger

from .config import get_settings


def setup_logger(
    log_file: Path | None = None,
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Set up the application logger with custom configuration.

    Args:
        log_file: Path to log file (defaults to settings.log_file)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files (e.g., "10 MB", "1 day")
        retention: How long to keep old log files (e.g., "1 week")
    """
    settings = get_settings()

    # Use settings if not provided
    if log_file is None:
        log_file = settings.log_file
    if log_level == "INFO":
        log_level = settings.log_level

    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler with custom format
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    # File handler with rotation
    logger.add(
        log_file,
        format=("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"),
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    logger.info(f"Logger initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> "logger":
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module (usually __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Initialize logger on import
setup_logger()

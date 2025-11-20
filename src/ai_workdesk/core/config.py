"""
Configuration management for AI Workdesk.

This module handles loading configuration from environment variables
and providing a centralized settings object using Pydantic Settings.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =============================================================================
    # LLM API KEYS
    # =============================================================================
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_org_id: Optional[str] = Field(None, description="OpenAI organization ID")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(None, description="Google AI API key")
    cohere_api_key: Optional[str] = Field(None, description="Cohere API key")

    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    default_llm_model: str = Field("gpt-4o-mini", description="Default LLM model")
    default_embedding_model: str = Field(
        "text-embedding-3-small", description="Default embedding model"
    )
    default_temperature: float = Field(0.7, ge=0.0, le=2.0, description="Default temperature")
    max_tokens: int = Field(4096, gt=0, description="Maximum tokens for generation")

    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    environment: Literal["development", "staging", "production"] = Field(
        "development", description="Application environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level"
    )
    log_file: Path = Field(Path("logs/ai_workdesk.log"), description="Log file path")

    # =============================================================================
    # VECTOR STORE CONFIGURATION
    # =============================================================================
    chroma_persist_directory: Path = Field(
        Path("data/chroma_db"), description="ChromaDB persist directory"
    )
    chroma_collection_name: str = Field(
        "ai_workdesk_collection", description="ChromaDB collection name"
    )

    # =============================================================================
    # DATA PATHS
    # =============================================================================
    raw_data_path: Path = Field(Path("data/raw"), description="Raw data directory")
    processed_data_path: Path = Field(
        Path("data/processed"), description="Processed data directory"
    )
    output_data_path: Path = Field(Path("data/outputs"), description="Output data directory")

    # =============================================================================
    # SECURITY & AUTHENTICATION
    # =============================================================================
    rate_limit_requests_per_minute: int = Field(
        60, gt=0, description="API rate limit (requests per minute)"
    )
    request_timeout: int = Field(30, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum number of retries")
    retry_delay: int = Field(1, ge=0, description="Retry delay in seconds")

    @field_validator("log_file", "chroma_persist_directory", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Ensure path fields are Path objects."""
        return Path(v) if isinstance(v, str) else v

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.log_file.parent,
            self.chroma_persist_directory,
            self.raw_data_path,
            self.processed_data_path,
            self.output_data_path,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_api_keys(self) -> dict[str, bool]:
        """Validate which API keys are configured."""
        return {
            "openai": self.openai_api_key is not None,
            "anthropic": self.anthropic_api_key is not None,
            "google": self.google_api_key is not None,
            "cohere": self.cohere_api_key is not None,
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(force_reload: bool = False) -> Settings:
    """
    Get the global settings instance.

    Args:
        force_reload: If True, reload settings from environment

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None or force_reload:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


# Convenience function
settings = get_settings()

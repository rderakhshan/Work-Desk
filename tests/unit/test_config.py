"""
Test configuration module.
"""

from pathlib import Path

from ai_workdesk.core.config import Settings, get_settings


def test_settings_default_values():
    """Test that settings have correct default values."""
    settings = Settings()

    assert settings.default_llm_model == "gpt-4o-mini"
    assert settings.default_embedding_model == "text-embedding-3-small"
    assert settings.default_temperature == 0.7
    assert settings.max_tokens == 4096
    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_settings_from_env(monkeypatch):
    """Test that settings can be loaded from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gpt-4")
    monkeypatch.setenv("DEFAULT_TEMPERATURE", "0.5")

    settings = Settings()

    assert settings.openai_api_key == "test-key-123"
    assert settings.default_llm_model == "gpt-4"
    assert settings.default_temperature == 0.5


def test_settings_path_validation():
    """Test that Path fields are properly validated."""
    settings = Settings(
        log_file="logs/test.log",
        raw_data_path="data/raw",
    )

    assert isinstance(settings.log_file, Path)
    assert isinstance(settings.raw_data_path, Path)
    assert settings.log_file == Path("logs/test.log")
    assert settings.raw_data_path == Path("data/raw")


def test_validate_api_keys(monkeypatch):
    """Test API key validation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings()
    validation = settings.validate_api_keys()

    assert validation["openai"] is True
    assert validation["anthropic"] is False
    assert validation["google"] is False
    assert validation["cohere"] is False


def test_get_settings_singleton():
    """Test that get_settings returns the same instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2


def test_get_settings_force_reload():
    """Test that force_reload creates a new instance."""
    settings1 = get_settings()
    settings2 = get_settings(force_reload=True)

    # Should be different instances
    assert settings1 is not settings2

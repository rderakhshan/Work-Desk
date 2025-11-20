"""
Test authentication module.
"""

import pytest

from ai_workdesk.core.auth import AuthenticationManager, get_auth_manager


def test_auth_manager_initialization():
    """Test that AuthenticationManager initializes correctly."""
    auth_manager = AuthenticationManager()
    assert auth_manager is not None
    assert auth_manager.settings is not None


def test_get_auth_manager_singleton():
    """Test that get_auth_manager returns the same instance."""
    manager1 = get_auth_manager()
    manager2 = get_auth_manager()

    assert manager1 is manager2


def test_check_authentication_status():
    """Test authentication status checking."""
    auth_manager = AuthenticationManager()
    status = auth_manager.check_authentication_status()

    assert isinstance(status, dict)
    assert "OpenAI" in status
    assert "Anthropic" in status
    assert "Google AI" in status
    assert "Cohere" in status


def test_validate_service_valid():
    """Test service validation with valid service names."""
    auth_manager = AuthenticationManager()

    # These should not raise errors
    auth_manager.validate_service("openai")
    auth_manager.validate_service("anthropic")
    auth_manager.validate_service("google")
    auth_manager.validate_service("cohere")


def test_validate_service_invalid():
    """Test service validation with invalid service name."""
    auth_manager = AuthenticationManager()

    with pytest.raises(ValueError, match="Unknown service"):
        auth_manager.validate_service("invalid_service")


def test_api_key_getters_with_env(monkeypatch):
    """Test API key getters with environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    auth_manager = AuthenticationManager()

    assert auth_manager.get_openai_api_key() == "test-openai-key"
    assert auth_manager.get_anthropic_api_key() == "test-anthropic-key"


def test_api_key_getters_without_env():
    """Test API key getters without environment variables."""
    auth_manager = AuthenticationManager()

    # These might be None if not configured
    openai_key = auth_manager.get_openai_api_key()
    anthropic_key = auth_manager.get_anthropic_api_key()

    # Keys are either string or None
    assert openai_key is None or isinstance(openai_key, str)
    assert anthropic_key is None or isinstance(anthropic_key, str)

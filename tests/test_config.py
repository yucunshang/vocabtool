# Tests for config.get_config model/default behavior.

import constants
import config as config_module


def test_get_config_uses_defaults_when_env_and_secrets_missing(monkeypatch):
    monkeypatch.setattr(config_module, "_get_secrets", lambda: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_DISPLAY", raising=False)

    cfg = config_module.get_config()

    assert cfg["openai_api_key"] == ""
    assert cfg["openai_base_url"] == constants.DEFAULT_OPENAI_BASE_URL
    assert cfg["openai_model"] == constants.DEFAULT_OPENAI_MODEL
    assert cfg["openai_model_display"] == constants.DEFAULT_OPENAI_MODEL_DISPLAY


def test_get_config_openai_model_display_can_be_overridden(monkeypatch):
    monkeypatch.setattr(config_module, "_get_secrets", lambda: {})
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENAI_MODEL_DISPLAY", "My Custom Label")

    cfg = config_module.get_config()

    assert cfg["openai_model"] == "deepseek-chat"
    assert cfg["openai_model_display"] == "My Custom Label"


def test_get_config_infers_display_for_custom_model(monkeypatch):
    monkeypatch.setattr(config_module, "_get_secrets", lambda: {})
    monkeypatch.setenv("OPENAI_MODEL", "my-private-model")
    monkeypatch.delenv("OPENAI_MODEL_DISPLAY", raising=False)

    cfg = config_module.get_config()

    assert cfg["openai_model"] == "my-private-model"
    assert cfg["openai_model_display"] == "my-private-model"

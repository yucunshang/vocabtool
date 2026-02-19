# Single config layer: Streamlit secrets first, then env, then defaults.

import os


def _get_secrets():
    try:
        import streamlit as st
        return getattr(st, "secrets", None) or {}
    except Exception:
        return {}


def get_config():
    """Return app config: st.secrets > env vars > defaults. Use in Streamlit context."""
    s = _get_secrets()
    return {
        "openai_api_key": s.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        "openai_base_url": s.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "openai_model": s.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL", "deepseek-chat"),
        "tts_provider": s.get("TTS_PROVIDER") or os.environ.get("TTS_PROVIDER", "edge"),
        "google_application_credentials": s.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        "google_tts_api_key": s.get("GOOGLE_TTS_API_KEY") or os.environ.get("GOOGLE_TTS_API_KEY", ""),
    }

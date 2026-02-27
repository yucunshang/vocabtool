# Single config layer: Streamlit secrets first, then env, then defaults.

import os

import constants


def _get_secrets():
    try:
        import streamlit as st
        return getattr(st, "secrets", None) or {}
    except Exception:
        return {}


def _infer_model_display(model_name: str) -> str:
    model = (model_name or "").strip()
    if not model:
        return constants.DEFAULT_OPENAI_MODEL_DISPLAY

    lower = model.lower()
    exact_map = {
        constants.DEFAULT_OPENAI_MODEL.lower(): constants.DEFAULT_OPENAI_MODEL_DISPLAY,
        "deepseek-reasoner": "DeepSeek-R1",
        "gpt-4o-mini": "OpenAI",
        "gpt-4o": "OpenAI",
    }
    if lower in exact_map:
        return exact_map[lower]
    if "deepseek" in lower:
        return "DeepSeek"
    if lower.startswith("gpt-") or lower.startswith("o1") or lower.startswith("o3"):
        return "OpenAI"
    return model


def get_config():
    """Return app config: st.secrets > env vars > defaults. Use in Streamlit context."""
    s = _get_secrets()
    openai_model = s.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL", constants.DEFAULT_OPENAI_MODEL)
    openai_model_display = (
        s.get("OPENAI_MODEL_DISPLAY")
        or os.environ.get("OPENAI_MODEL_DISPLAY", "").strip()
        or _infer_model_display(openai_model)
    )

    return {
        "openai_api_key": s.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        "openai_base_url": s.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", constants.DEFAULT_OPENAI_BASE_URL),
        "openai_model": openai_model,
        "openai_model_display": openai_model_display,
    }

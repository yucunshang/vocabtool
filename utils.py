# Shared utility functions.

import gc
from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

import constants

logger = __import__("logging").getLogger(__name__)


def safe_str_clean(value: Any) -> str:
    """Convert to string and clean whitespace safely."""
    if value is None:
        return ""
    return str(value).strip()


def render_copy_button(text: str, key: str = "copy_words") -> None:
    """Render a compatibility-safe export button (fallback for old browsers)."""
    st.download_button(
        label="📥",
        data=text,
        file_name="word_list.txt",
        mime="text/plain; charset=utf-8",
        key=f"{key}_download",
        use_container_width=True,
        help="下载当前词表（用于手动复制）",
    )


def render_prompt_copy_button(text: str, key: str = "copy_prompt") -> None:
    """Render a compatibility-safe prompt export button."""
    st.download_button(
        label="📥 下载 Prompt",
        data=text,
        file_name="prompt.txt",
        mime="text/plain; charset=utf-8",
        key=f"{key}_download",
        use_container_width=False,
    )


def run_gc() -> None:
    """Run garbage collection to reduce memory pressure after heavy tasks."""
    try:
        gc.collect()
    except Exception:
        pass


def get_beijing_time_str() -> str:
    """Get current Beijing time as formatted string."""
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=constants.BEIJING_TIMEZONE_OFFSET)
    return beijing_now.strftime('%m%d_%H%M')


def detect_file_encoding(bytes_data: bytes) -> str:
    """Detect file encoding using chardet if available, fallback to priority list."""
    try:
        import chardet
        detected = chardet.detect(bytes_data)
        encoding = detected.get('encoding')
        if encoding and detected.get('confidence', 0) > 0.7:
            return encoding
    except ImportError:
        logger.debug("chardet not available, using fallback encodings")
    for encoding in constants.ENCODING_PRIORITY:
        try:
            bytes_data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return 'latin-1'

# Session state helpers.

import json
from typing import Dict, List, Optional, Tuple

import streamlit as st


def set_generated_words_state(
    data_list: List[Tuple[str, int]],
    raw_count: int = 0,
    stats_info: Optional[Dict[str, float]] = None
) -> None:
    """Update extracted words and keep editor text in sync with new generation."""
    st.session_state['gen_words_data'] = data_list
    st.session_state['raw_count'] = raw_count
    st.session_state['stats_info'] = stats_info
    st.session_state['word_list_editor'] = "\n".join([w for w, _ in data_list])


def export_session_json() -> str:
    """Serialize exportable session state to a JSON string."""
    data = {
        "version": 1,
        "gen_words_data": [list(t) for t in (st.session_state.get("gen_words_data") or [])],
        "raw_count": st.session_state.get("raw_count", 0),
        "stats_info": st.session_state.get("stats_info"),
        "word_list_editor": st.session_state.get("word_list_editor", ""),
        "anki_cards_cache": st.session_state.get("anki_cards_cache"),
        "anki_pkg_name": st.session_state.get("anki_pkg_name", ""),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def import_session_json(json_str: str) -> None:
    """Restore session state from a JSON string produced by export_session_json."""
    data = json.loads(json_str)
    if data.get("version") != 1:
        raise ValueError("Unsupported session file version")
    st.session_state["gen_words_data"] = [tuple(t) for t in data.get("gen_words_data", [])]
    st.session_state["raw_count"] = data.get("raw_count", 0)
    st.session_state["stats_info"] = data.get("stats_info")
    st.session_state["word_list_editor"] = data.get("word_list_editor", "")
    st.session_state["anki_cards_cache"] = data.get("anki_cards_cache")
    st.session_state["anki_pkg_name"] = data.get("anki_pkg_name", "")

# Session state helpers.

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

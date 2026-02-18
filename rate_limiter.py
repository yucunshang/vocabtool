# Session-scoped rate limiter using rolling time windows.
#
# Each Streamlit session gets its own counters stored in st.session_state.
# This protects against one user abusing the API while keeping thresholds
# high enough that normal human usage is never affected.

import time
from collections import deque
from typing import Tuple

import streamlit as st

import constants


def _get_timestamps(key: str) -> deque:
    """Return the rolling timestamp deque for *key*, creating it if absent."""
    if key not in st.session_state:
        st.session_state[key] = deque()
    return st.session_state[key]


def _prune(timestamps: deque, window_seconds: float) -> deque:
    """Remove entries older than *window_seconds* from the deque (in-place)."""
    cutoff = time.time() - window_seconds
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()
    return timestamps


def check_rate_limit(
    action: str,
    per_minute: int,
    per_hour: int,
    per_day: int,
) -> Tuple[bool, str]:
    """Check whether *action* is within rate limits.

    Returns ``(allowed, message)``.  If *allowed* is False, *message*
    contains a user-friendly Chinese explanation.
    """
    key_min = f"_rl_{action}_min"
    key_hour = f"_rl_{action}_hour"
    key_day = f"_rl_{action}_day"

    ts_min = _prune(_get_timestamps(key_min), 60)
    ts_hour = _prune(_get_timestamps(key_hour), 3600)
    ts_day = _prune(_get_timestamps(key_day), 86400)

    if len(ts_min) >= per_minute:
        return False, f"⚠️ 操作过于频繁（每分钟上限 {per_minute} 次），请稍后再试。"
    if len(ts_hour) >= per_hour:
        return False, f"⚠️ 已达到小时上限（{per_hour} 次/小时），请稍后再试。"
    if len(ts_day) >= per_day:
        return False, f"⚠️ 已达到每日上限（{per_day} 次/天），明天再来吧。"

    return True, ""


def record_usage(action: str) -> None:
    """Record one successful usage event for *action*."""
    now = time.time()
    for suffix in ("_min", "_hour", "_day"):
        key = f"_rl_{action}{suffix}"
        if key not in st.session_state:
            st.session_state[key] = deque()
        st.session_state[key].append(now)


# ---------- Convenience wrappers ----------

def check_lookup_limit() -> Tuple[bool, str]:
    """Rate-check for AI word lookup."""
    return check_rate_limit(
        "lookup",
        per_minute=constants.RL_LOOKUP_PER_MINUTE,
        per_hour=constants.RL_LOOKUP_PER_HOUR,
        per_day=constants.RL_LOOKUP_PER_DAY,
    )


def record_lookup() -> None:
    record_usage("lookup")


def check_batch_limit() -> Tuple[bool, str]:
    """Rate-check for batch AI card generation."""
    return check_rate_limit(
        "batch",
        per_minute=constants.RL_BATCH_PER_MINUTE,
        per_hour=constants.RL_BATCH_PER_HOUR,
        per_day=constants.RL_BATCH_PER_DAY,
    )


def record_batch() -> None:
    record_usage("batch")


def check_url_limit() -> Tuple[bool, str]:
    """Rate-check for URL scraping."""
    return check_rate_limit(
        "url",
        per_minute=constants.RL_URL_PER_MINUTE,
        per_hour=constants.RL_URL_PER_HOUR,
        per_day=constants.RL_URL_PER_DAY,
    )


def record_url() -> None:
    record_usage("url")

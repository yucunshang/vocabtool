# Tests for ai.build_card_prompt (no API calls).
# Uses fixed built-in template; fmt parameter is ignored.

import pytest

from ai import DEFAULT_CARD_FORMAT, build_card_prompt


def test_build_card_prompt_contains_separator():
    out = build_card_prompt("hello, world", DEFAULT_CARD_FORMAT)
    assert "|||" in out


def test_build_card_prompt_contains_word_list():
    out = build_card_prompt("altruism, hectic", None)
    assert "altruism" in out
    assert "hectic" in out


def test_build_card_prompt_fixed_template_has_field_constraints():
    """Template includes Target Word, definition, example (3 fields, no etymology)."""
    out = build_card_prompt("test", None)
    assert "Target Word" in out
    assert "3 fields" in out
    assert "|||" in out


def test_build_card_prompt_batch_limit_constraint():
    """Fixed template enforces 10-word batch limit."""
    out = build_card_prompt("word1", None)
    assert "10 words" in out or "10-word" in out


def test_build_card_prompt_no_etymology():
    """Minimalist template explicitly excludes etymology/roots."""
    out = build_card_prompt("boycott", None)
    assert "NO ETYMOLOGY" in out or "no etymology" in out or "Do NOT output any etymology" in out


def test_build_card_prompt_chinese_only_minimalist():
    """Minimalist template uses Chinese-only definition, one example."""
    out = build_card_prompt("test", None)
    assert "中文" in out or "Chinese" in out
    assert "ONE" in out or "one" in out  # one example
    assert "|||" in out


def test_build_card_prompt_empty_words():
    out = build_card_prompt("", DEFAULT_CARD_FORMAT)
    assert "|||" in out
    assert isinstance(out, str)
    assert len(out) > 0

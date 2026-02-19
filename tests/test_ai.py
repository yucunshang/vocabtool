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
    """Template includes Word/Phrase, Definition, Examples (and optionally Etymology)."""
    out = build_card_prompt("test", None)
    assert "Field 1:" in out and ("Word" in out or "Phrase" in out)
    assert "Field 2:" in out
    assert "Field 3:" in out
    assert "|||" in out


def test_build_card_prompt_batch_limit_constraint():
    """Fixed template enforces 10-word batch limit."""
    out = build_card_prompt("word1", None)
    assert "10 words" in out or "10-word" in out


def test_build_card_prompt_etymology_zero_hallucination():
    """Template includes 词源不可考 when etymology is enabled."""
    fmt = {"front": "word", "definition": "both", "examples": 2, "etymology": True, "examples_with_cn": True}
    out = build_card_prompt("boycott", fmt)
    assert "词源不可考" in out


def test_build_card_prompt_fixed_bilingual_and_etymology():
    """Fixed template requires bilingual definition and includes deep etymology rules (fmt ignored)."""
    out = build_card_prompt("test", None)
    assert "中文释义" in out or "Definition (Bilingual)" in out
    assert "Etymology" in out or "词源" in out
    assert "|||" in out


def test_build_card_prompt_empty_words():
    out = build_card_prompt("", DEFAULT_CARD_FORMAT)
    assert "|||" in out
    assert isinstance(out, str)
    assert len(out) > 0

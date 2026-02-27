# Tests for ai.build_card_prompt (no API calls).
# Uses fixed built-in template; fmt parameter is ignored.

import pytest

from ai import (
    DEFAULT_CARD_FORMAT,
    build_card_prompt,
    build_thirdparty_format_definition,
    build_thirdparty_prompt,
)


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
    """Fixed template enforces batch limit (10 words)."""
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


def test_build_thirdparty_prompt_standard_strong_model_template():
    out = build_thirdparty_prompt("altruism, hectic", {"card_type": "standard"})
    assert "strong frontier models" in out
    assert "|||" in out
    assert "no skipping, no merging" in out


def test_build_thirdparty_prompt_cloze_not_limited_to_10():
    out = build_thirdparty_prompt("brass, dam", {"card_type": "cloze", "voice_code": "en-US-JennyNeural"})
    assert "Expert Lexicographer & Reading Card Generator for Strong LLMs." in out
    assert "Target word /IPA/ pos. 中文释义" in out
    assert "Max 10" not in out and "max 10" not in out
    assert "|||" in out


def test_build_thirdparty_format_definition_for_new_card_types():
    out = build_thirdparty_format_definition({"card_type": "translation"})
    assert "总量不限" in out
    assert "互译卡" in out
    assert "|||" in out

# Tests for ai.build_card_prompt (no API calls).

import pytest

from ai import DEFAULT_CARD_FORMAT, build_card_prompt


def test_build_card_prompt_contains_separator():
    out = build_card_prompt("hello, world", DEFAULT_CARD_FORMAT)
    assert "|||" in out


def test_build_card_prompt_contains_word_list():
    out = build_card_prompt("altruism, hectic", None)
    assert "altruism" in out
    assert "hectic" in out


def test_build_card_prompt_default_format_word_cn():
    fmt = {
        "front": "word",
        "definition": "cn",
        "examples": 2,
        "examples_with_cn": True,
        "etymology": False,
    }
    out = build_card_prompt("test", fmt)
    assert "Field 1: Word" in out or "word" in out.lower()
    assert "中文" in out or "Chinese" in out


def test_build_card_prompt_phrase_format():
    fmt = {"front": "phrase", "definition": "en", "examples": 1, "etymology": False}
    out = build_card_prompt("rain", fmt)
    assert "Phrase" in out or "phrase" in out.lower()
    assert "collocation" in out.lower() or "phrase" in out.lower()


def test_build_card_prompt_empty_words():
    out = build_card_prompt("", DEFAULT_CARD_FORMAT)
    assert "|||" in out
    assert isinstance(out, str)
    assert len(out) > 0

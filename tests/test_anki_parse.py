# Tests for anki_parse.parse_anki_data.

import pytest

from anki_parse import parse_anki_data


def test_parse_anki_data_empty():
    assert parse_anki_data("") == []
    assert parse_anki_data("   \n  ") == []


def test_parse_anki_data_single_line():
    raw = "hello ||| 你好 ||| Hello world. ||| 词源"
    result = parse_anki_data(raw)
    assert len(result) == 1
    assert result[0]["w"] == "hello"
    assert result[0]["m"] == "你好"
    assert result[0]["e"] == "Hello world."
    assert result[0]["r"] == "词源"


def test_parse_anki_data_two_fields():
    raw = "word ||| meaning only"
    result = parse_anki_data(raw)
    assert len(result) == 1
    assert result[0]["w"] == "word"
    assert result[0]["m"] == "meaning only"
    assert result[0]["e"] == ""
    assert result[0]["r"] == ""


def test_parse_anki_data_code_block():
    raw = """Some text before
```text
hectic ||| 忙乱的 ||| She has a hectic day. ||| hect- + -ic
altruism ||| 利他 ||| Motivated by altruism. ||| alter + -ism
```
After."""
    result = parse_anki_data(raw)
    assert len(result) == 2
    assert result[0]["w"] == "hectic"
    assert result[0]["m"] == "忙乱的"
    assert result[1]["w"] == "altruism"
    assert result[1]["m"] == "利他"


def test_parse_anki_data_deduplicates():
    raw = """word ||| meaning ||| ex ||| ety
word ||| other meaning ||| ex2 ||| ety2"""
    result = parse_anki_data(raw)
    assert len(result) == 1
    assert result[0]["m"] == "meaning"


def test_parse_anki_data_skips_invalid():
    raw = """ ||| meaning ||| ex ||| ety
word |||  ||| ex ||| ety
valid ||| ok ||| ex ||| ety"""
    result = parse_anki_data(raw)
    assert len(result) == 1
    assert result[0]["w"] == "valid"


def test_parse_anki_data_block_format_multiline():
    """Block format: blank line between cards; 5-field cloze (word/IPA, 释义, 搭配, example)."""
    raw = """The terms were so ________ that both sides disagreed. ||| ambiguous / æmˈbɪɡjuəs ||| 模糊的；unclear, having multiple meanings ||| ambiguous statement ||| The terms were so ambiguous that both sides disagreed. |||

Another ________ example. ||| word2 / w2 ||| 释义；definition ||| collocation ||| Another word2 example. |||"""
    result = parse_anki_data(raw)
    assert len(result) == 2
    assert result[0]["w"] == "The terms were so ________ that both sides disagreed."
    assert "ambiguous" in result[0]["m"] and "模糊的" in result[0]["m"] and "ambiguous statement" in result[0]["m"]
    assert result[0]["e"] == "The terms were so ambiguous that both sides disagreed."
    assert result[1]["w"] == "Another ________ example."
    assert "word2" in result[1]["m"] and "释义" in result[1]["m"]


def test_parse_anki_data_cloze_c1_format():
    """5-field cloze with {{c1::word}} format."""
    raw = """The doorknob, made of polished {{c1::brass}}, gleamed in the light. ||| brass / bræs ||| n. 黄铜 ||| polished brass ||| The doorknob, made of polished brass, gleamed in the light."""
    result = parse_anki_data(raw)
    assert len(result) == 1
    assert "{{c1::brass}}" in result[0]["w"]
    assert "brass" in result[0]["m"]
    assert result[0]["e"] == "The doorknob, made of polished brass, gleamed in the light."


def test_parse_anki_data_card_shape():
    raw = "phrase ||| def ||| example ||| etymology"
    result = parse_anki_data(raw)
    assert len(result) == 1
    card = result[0]
    assert set(card.keys()) == {"w", "m", "e", "r"}
    assert card["w"] == "phrase"
    assert card["m"] == "def"
    assert card["e"] == "example"
    assert card["r"] == "etymology"

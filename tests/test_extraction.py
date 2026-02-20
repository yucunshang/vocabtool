# Tests for extraction.clean_anki_field and parse_anki_txt_export.

import socket
from io import BytesIO

import pytest

from extraction import _is_safe_url, clean_anki_field, parse_anki_txt_export


def test_clean_anki_field_plain():
    assert clean_anki_field("hello") == "hello"
    assert clean_anki_field("  word  ") == "word"


def test_clean_anki_field_cloze():
    assert clean_anki_field("{{c1::answer}}") == "answer"
    assert clean_anki_field("{{c2::word::hint}}") == "word"


def test_clean_anki_field_html():
    # Tags removed and result stripped
    assert clean_anki_field("<b>bold</b>") == "bold"
    assert clean_anki_field("<div>text</div>") == "text"


def test_clean_anki_field_sound():
    assert clean_anki_field("[sound:file.mp3]") == ""
    assert clean_anki_field("word [sound:a.mp3]") == "word"


def test_clean_anki_field_unescape():
    assert "&" in clean_anki_field("a & b")
    # After unescape we get "a & b" (single char for &)


def test_parse_anki_txt_export_simple_tsv():
    content = b"word1\nword2\nword3"
    f = BytesIO(content)
    f.name = "export.txt"
    result = parse_anki_txt_export(f)
    lines = result.strip().split("\n")
    assert len(lines) == 3
    assert "word1" in lines and "word2" in lines and "word3" in lines


def test_parse_anki_txt_export_skips_short():
    content = b"a\nab\nabc"
    f = BytesIO(content)
    f.name = "export.txt"
    result = parse_anki_txt_export(f)
    lines = [x for x in result.strip().split("\n") if x]
    assert "a" not in lines
    assert "ab" in lines
    assert "abc" in lines


def test_parse_anki_txt_export_skips_comments():
    content = b"# comment\nword1\nword2"
    f = BytesIO(content)
    f.name = "export.txt"
    result = parse_anki_txt_export(f)
    assert "comment" not in result
    assert "word1" in result


def test_is_safe_url_blocks_localhost():
    assert _is_safe_url("http://localhost/test") is False
    assert _is_safe_url("https://127.0.0.1/") is False


def test_is_safe_url_blocks_private_resolution(monkeypatch):
    def _fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.5", 0))]

    monkeypatch.setattr("extraction.socket.getaddrinfo", _fake_getaddrinfo)
    assert _is_safe_url("https://example.com/path") is False


def test_is_safe_url_allows_public_resolution(monkeypatch):
    def _fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 0))]

    monkeypatch.setattr("extraction.socket.getaddrinfo", _fake_getaddrinfo)
    assert _is_safe_url("https://example.com/path") is True

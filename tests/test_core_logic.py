# Tests for load_vocab_data (resources.py) and get_openai_client (ai.py).
# Both modules are decoupled from Streamlit and must be testable without a UI context.

import os

import pandas as pd
import pytest

import ai as ai_module
import resources


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(data: dict) -> pd.DataFrame:
    """Build a minimal word/rank DataFrame from {word: rank}."""
    return pd.DataFrame({"word": list(data.keys()), "rank": list(data.values())})


# ---------------------------------------------------------------------------
# resources.load_vocab_data
# ---------------------------------------------------------------------------

class TestLoadVocabData:
    """load_vocab_data() must work without Streamlit and return consistent types."""

    def setup_method(self):
        resources.load_vocab_data.cache_clear()

    def teardown_method(self):
        resources.load_vocab_data.cache_clear()

    def test_returns_two_tuple(self):
        """Return value is always a 2-tuple regardless of whether a CSV exists."""
        result = resources.load_vocab_data()
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_dict(self):
        """First element of the tuple is always a dict."""
        vocab, _ = resources.load_vocab_data()
        assert isinstance(vocab, dict)

    def test_second_element_is_df_or_none(self):
        """Second element is a DataFrame or None."""
        _, df = resources.load_vocab_data()
        assert df is None or isinstance(df, pd.DataFrame)

    def test_no_csv_returns_empty(self, monkeypatch):
        """When no CSV file is found, returns ({}, None) without raising."""
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        vocab, df = resources.load_vocab_data()
        assert vocab == {}
        assert df is None

    def test_parses_word_and_rank_columns(self, monkeypatch):
        """A valid CSV is parsed into a {word: rank} dict and a DataFrame."""
        source = {"hectic": 8000, "altruism": 12000, "serendipity": 18000}
        test_df = _make_df(source)

        monkeypatch.setattr(os.path, "exists", lambda p: p.endswith("ngsl_word_rank.csv"))
        monkeypatch.setattr(resources.pd, "read_csv", lambda _p: test_df.copy())

        vocab, df = resources.load_vocab_data()

        assert vocab["hectic"] == 8000
        assert vocab["altruism"] == 12000
        assert vocab["serendipity"] == 18000
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_result_is_cached(self, monkeypatch):
        """Calling load_vocab_data() twice returns the exact same object (lru_cache)."""
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        first = resources.load_vocab_data()
        second = resources.load_vocab_data()
        assert first is second

    def test_csv_read_error_returns_empty(self, monkeypatch):
        """If reading the CSV raises, gracefully returns ({}, None)."""
        def _raise(_p):
            raise OSError("simulated disk error")

        monkeypatch.setattr(os.path, "exists", lambda p: p.endswith("ngsl_word_rank.csv"))
        monkeypatch.setattr(resources.pd, "read_csv", _raise)

        vocab, df = resources.load_vocab_data()
        assert vocab == {}
        assert df is None

    def test_duplicate_words_kept_lowest_rank(self, monkeypatch):
        """When a word appears multiple times, only the lowest rank entry is kept."""
        test_df = pd.DataFrame({
            "word": ["run", "run", "run"],
            "rank": [500.0, 200.0, 800.0],
        })
        monkeypatch.setattr(os.path, "exists", lambda p: p.endswith("ngsl_word_rank.csv"))
        monkeypatch.setattr(resources.pd, "read_csv", lambda _p: test_df.copy())

        vocab, _ = resources.load_vocab_data()
        assert vocab["run"] == 200


# ---------------------------------------------------------------------------
# resources.get_vocab_dict
# ---------------------------------------------------------------------------

class TestGetVocabDict:
    """get_vocab_dict() must delegate to load_vocab_data() with no extra logic."""

    def test_returns_same_object_as_load_vocab_data(self, monkeypatch):
        """get_vocab_dict() returns the identical dict object from load_vocab_data()."""
        mock_vocab = {"hello": 500, "world": 800}
        monkeypatch.setattr(resources, "load_vocab_data", lambda: (mock_vocab, None))
        assert resources.get_vocab_dict() is mock_vocab

    def test_returns_dict_type(self, monkeypatch):
        monkeypatch.setattr(resources, "load_vocab_data", lambda: ({}, None))
        assert isinstance(resources.get_vocab_dict(), dict)


# ---------------------------------------------------------------------------
# resources.get_rank_for_word
# ---------------------------------------------------------------------------

class TestGetRankForWord:
    """get_rank_for_word() must resolve ranks correctly without touching the cache."""

    _VOCAB = {"hectic": 8000, "hello": 500, "May": 300}

    @pytest.fixture(autouse=True)
    def patch_vocab(self, monkeypatch):
        monkeypatch.setattr(resources, "load_vocab_data", lambda: (self._VOCAB, None))

    def test_exact_match(self):
        assert resources.get_rank_for_word("hectic") == 8000

    def test_exact_case_takes_priority_over_lowercase(self):
        """'May' (proper noun) should match before its lowercase form."""
        assert resources.get_rank_for_word("May") == 300

    def test_lowercase_fallback(self):
        """If exact case not found, the lowercase form is tried."""
        assert resources.get_rank_for_word("HELLO") == 500
        assert resources.get_rank_for_word("Hello") == 500

    def test_unknown_word_returns_99999(self):
        assert resources.get_rank_for_word("xyzzy") == 99999

    def test_empty_string_returns_99999(self):
        assert resources.get_rank_for_word("") == 99999

    def test_whitespace_is_stripped(self):
        assert resources.get_rank_for_word("  hectic  ") == 8000

    def test_none_like_empty_input(self):
        """Single space after strip is empty — treated as not found."""
        assert resources.get_rank_for_word("   ") == 99999


# ---------------------------------------------------------------------------
# ai.get_openai_client — no actual API calls made
# ---------------------------------------------------------------------------

class TestGetOpenAIClient:
    """get_openai_client() must handle missing library, missing key, and happy path."""

    @pytest.fixture(autouse=True)
    def reset_client_cache(self, monkeypatch):
        """Reset the module-level _OPENAI_CLIENT before every test."""
        monkeypatch.setattr(ai_module, "_OPENAI_CLIENT", None)

    def test_returns_none_when_openai_not_installed(self, monkeypatch):
        """If the OpenAI library is absent (OpenAI = None), returns None without raising."""
        monkeypatch.setattr(ai_module, "OpenAI", None)
        assert ai_module.get_openai_client() is None

    def test_returns_none_when_api_key_empty(self, monkeypatch):
        """If OPENAI_API_KEY is an empty string, returns None."""
        class _FakeLib:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(ai_module, "OpenAI", _FakeLib)
        monkeypatch.setattr(
            "ai.get_config",
            lambda: {
                "openai_api_key": "",
                "openai_base_url": "https://api.openai.com/v1",
                "openai_model": "test-model",
            },
        )
        assert ai_module.get_openai_client() is None

    def test_returns_client_when_configured(self, monkeypatch):
        """With a valid API key and library, returns an initialized client instance."""
        class _FakeLib:
            def __init__(self, api_key, base_url, timeout):
                self.api_key = api_key
                self.base_url = base_url

        monkeypatch.setattr(ai_module, "OpenAI", _FakeLib)
        monkeypatch.setattr(
            "ai.get_config",
            lambda: {
                "openai_api_key": "sk-test-key",
                "openai_base_url": "https://api.openai.com/v1",
                "openai_model": "test-model",
            },
        )

        client = ai_module.get_openai_client()

        assert client is not None
        assert isinstance(client, _FakeLib)
        assert client.api_key == "sk-test-key"
        assert client.base_url == "https://api.openai.com/v1"

    def test_client_is_cached_on_second_call(self, monkeypatch):
        """A second call returns the same client instance (module-level cache)."""
        class _FakeLib:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(ai_module, "OpenAI", _FakeLib)
        monkeypatch.setattr(
            "ai.get_config",
            lambda: {
                "openai_api_key": "sk-test-key",
                "openai_base_url": "https://api.openai.com/v1",
                "openai_model": "test-model",
            },
        )

        first = ai_module.get_openai_client()
        second = ai_module.get_openai_client()
        assert first is second

    def test_importable_without_streamlit(self):
        """ai module is loaded and functional without requiring streamlit in scope."""
        assert callable(ai_module.get_openai_client)
        assert callable(ai_module.build_card_prompt)
        assert callable(ai_module.process_ai_in_batches)

    def test_resources_importable_without_streamlit(self):
        """resources module is loaded and functional without requiring streamlit in scope."""
        assert callable(resources.load_vocab_data)
        assert callable(resources.get_vocab_dict)
        assert callable(resources.get_rank_for_word)


class TestRankFromAIContent:
    def test_skips_typo_notice_line_with_checkmark(self, monkeypatch):
        monkeypatch.setattr(ai_module, "get_rank_for_word", lambda w: 123 if w == "receive" else 99999)
        content = "✔️ 拼写纠正: recieve -> receive\nreceive (v. 动词)\n接收 | to get something"
        assert ai_module._rank_from_ai_content(content, 888) == 123

    def test_skips_typo_notice_line_with_pencil(self, monkeypatch):
        monkeypatch.setattr(ai_module, "get_rank_for_word", lambda w: 321 if w == "receive" else 99999)
        content = "✏️ 拼写纠正: recieve -> receive\nreceive (v. 动词)\n接收 | to get something"
        assert ai_module._rank_from_ai_content(content, 888) == 321

# CLAUDE.md — Architecture & Developer Guide

## Module Map

### Pure-logic modules (no Streamlit dependency)
These can be imported and tested without a running Streamlit session.

| Module | Role |
|---|---|
| `resources.py` | Cached loaders: vocab CSV, NLP (NLTK/lemminflect), file parsers (pypdf/docx/epub), genanki. Uses `functools.lru_cache` — no `st.cache_*`. |
| `ai.py` | OpenAI client init, word lookup (`get_word_quick_definition`), batch card generation (`process_ai_in_batches`), prompt building (`build_card_prompt`). Errors go to `logger.error`, not `st.error`. |
| `vocab.py` | Rank-based word extraction (`analyze_logic`), word validation. |
| `anki_parse.py` | Parse AI-returned `|||`-delimited text into card dicts. |
| `anki_package.py` | Build `.apkg` files via genanki + TTS audio. |
| `extraction.py` | Extract plain text from PDF, DOCX, EPUB, URL, CSV, Excel, SQLite, Anki export. |
| `tts.py` | Async batch TTS via edge-tts. |
| `constants.py` | App-wide constants (rank limits, batch sizes, voice map, etc.). |
| `config.py` | Read config from `st.secrets` → env vars → defaults. Gracefully skips Streamlit when absent. |
| `prompts.py` | Prompt templates (`CARD_GEN_SYSTEM_PROMPT`, `CARD_GEN_USER_TEMPLATE`, `LOOKUP_SYSTEM_PROMPT`). Edit here to change AI behaviour without touching logic. |

### UI-coupled modules (import streamlit)
| Module | Role |
|---|---|
| `app.py` | Streamlit entry point: page config, tabs, session-state wiring, UI rendering. |
| `errors.py` | `ErrorHandler` — shows `st.error` and logs. |
| `state.py` | Session-state helpers for generated word list. |
| `rate_limiter.py` | Per-session rate limits via `st.session_state`. |
| `utils.py` | `render_copy_button`, `run_gc`, `get_beijing_time_str`. |
| `ui_styles.py` | Inline CSS injected via `st.markdown`. |

---

## Key Architecture Decisions

### `resources.py`
- `load_vocab_data()` is the **single source of truth** for the vocab dict. It is `lru_cache`'d and called internally by `get_vocab_dict()` and `get_rank_for_word()`.
- No module-level mutable globals (`VOCAB_DICT`, `FULL_DF` were removed Feb 2026).
- `app.py` loads `VOCAB_DICT, FULL_DF = resources.load_vocab_data()` for the rank-browser tab and the startup error check only.

### `ai.py`
- `get_openai_client()` returns `None` (and logs) on missing library or missing API key. Callers in `app.py` already handle `None` with their own user-facing messages.
- Module-level `_OPENAI_CLIENT` caches the client after first successful init.

---

## `app.py` Helper Functions

`_render_extract_results()` (wrapped with `st.fragment`) delegates to these private helpers:

| Function | Signature | Purpose |
|---|---|---|
| `_render_extraction_stats` | `(data: list) -> None` | Show 4 metrics: coverage, target density, raw count, filtered count. |
| `_render_word_editor` | `(data: list) -> list` | Editable textarea with copy button. Returns the current word list (edited or original). |
| `_render_audio_settings` | `(key_prefix: str) -> tuple[bool, str, bool]` | Audio enable checkbox + voice radio. `key_prefix` keeps widget keys unique (use `"auto"` here). Returns `(enable_audio, voice_code, enable_example_audio)`. |
| `_render_builtin_ai_section` | `(words_only, enable_audio, voice_code, enable_example_audio, card_format) -> None` | Generate button, progress bars, AI batch call, package generation, download button. |

---

## Testing

- **Python:** `C:\Users\shangcunyu\AppData\Local\Programs\Python\Python313\python.exe`
- **Run:** `python -m pytest tests/ -v`
- **JUnit XML:** `python -m pytest tests/ --junitxml=test-results.xml` (committed to repo)
- **69 tests** across: `test_ai.py`, `test_anki_parse.py`, `test_config.py`, `test_core_logic.py`, `test_extraction.py`, `test_vocab.py`
- `test_core_logic.py` proves `resources.py` and `ai.py` are testable without Streamlit.

### Test patterns
- `lru_cache` functions: call `fn.cache_clear()` in `setup_method` and `teardown_method`.
- Control `load_vocab_data()` return value: `monkeypatch.setattr(resources, "load_vocab_data", lambda: (mock_vocab, None))`.
- Reset AI client cache: `monkeypatch.setattr(ai_module, "_OPENAI_CLIENT", None)`.
- Mock `get_config()`: `monkeypatch.setattr("ai.get_config", lambda: {...})`.

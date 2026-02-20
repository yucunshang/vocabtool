# Anki package (.apkg) generation with optional TTS.

import hashlib
import os
import tempfile
import time
import zlib
import re
from typing import Dict, List, Optional, Tuple

import constants
from errors import ProgressCallback
from resources import get_genanki
from tts import run_async_batch
from utils import safe_str_clean

APKG_TEMP_DIR = os.path.join(tempfile.gettempdir(), constants.APKG_TEMP_SUBDIR)
# Persistent, content-addressed audio cache — survives between generate_anki_package calls
# so that a retry only re-generates genuinely missing clips.
AUDIO_CACHE_DIR = os.path.join(APKG_TEMP_DIR, "audio_cache")


def _audio_cache_path(text: str, voice: str) -> str:
    """Return a stable file path keyed by (text, voice).

    Using MD5 of the key means the same word/sentence+voice always maps to the
    same filename, so a second call (retry) skips files that already exist.
    """
    key = f"{voice}:{text}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(AUDIO_CACHE_DIR, f"tts_{h}.mp3")


def cleanup_old_apkg_files(max_age_seconds: int = constants.APKG_CLEANUP_MAX_AGE_SECONDS) -> None:
    """Remove old .apkg files and stale cached TTS audio files."""
    now = time.time()
    # Clean .apkg output files
    if os.path.isdir(APKG_TEMP_DIR):
        try:
            for name in os.listdir(APKG_TEMP_DIR):
                if not name.endswith(".apkg"):
                    continue
                path = os.path.join(APKG_TEMP_DIR, name)
                if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age_seconds:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        except OSError:
            pass
    # Clean old cached TTS audio files
    if os.path.isdir(AUDIO_CACHE_DIR):
        try:
            for name in os.listdir(AUDIO_CACHE_DIR):
                if not name.endswith(".mp3"):
                    continue
                path = os.path.join(AUDIO_CACHE_DIR, name)
                if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age_seconds:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        except OSError:
            pass


def _split_examples(raw_example: str) -> List[str]:
    """Split a raw example field that may contain ``//``-separated sentences."""
    if not raw_example:
        return []
    parts = [s.strip() for s in re.split(r'\s*//\s*', raw_example) if s.strip()]
    return parts if parts else ([raw_example] if raw_example.strip() else [])


def _example_text_for_tts(display_text: str) -> str:
    """Return the text to send to TTS: only the English part when format is 'English (中文)'."""
    idx = display_text.find(" (")
    if idx != -1:
        return display_text[:idx].strip()
    return display_text.strip()


def generate_anki_package(
    cards_data: List[Dict[str, str]],
    deck_name: str,
    enable_tts: bool = False,
    tts_voice: str = "en-US-JennyNeural",
    enable_example_tts: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    card_type: str = "standard",
) -> Tuple[str, int, List[str]]:
    """Generate Anki package (.apkg) file with optional TTS audio.

    Returns:
        (apkg_path, failed_audio_count, failed_phrases): path to the generated file,
        the number of audio clips that could not be produced, and the list of
        phrase strings (words) for cards that had at least one audio failure.  On retry the
        persistent audio cache (AUDIO_CACHE_DIR) means only genuinely missing
        clips are re-requested from the TTS service.

    Handles variable card formats: examples may be ``//``-separated and
    etymology may be absent.  Audio is generated for the phrase and for
    each individual example sentence.
    """
    genanki, tempfile_mod = get_genanki()
    media_files: List[str] = []

    CSS = """
    .card {
        font-family: -apple-system, 'Arial', sans-serif;
        font-size: 22px; text-align: center; color: #333;
        background-color: white;
        padding: 14px 16px; margin: 0 auto;
    }
    .phrase {
        font-size: 30px; font-weight: 700; color: #0056b3;
        margin: 0 0 6px 0; line-height: 1.3;
    }
    .nightMode .card { background-color: #1a1a2e; color: #e0e0e0; }
    .nightMode .phrase { color: #7cb8ff; }

    .audio-phrase, .audio-ex {
        display: inline-block; vertical-align: middle;
        margin: 0 0 0 2px;
    }
    .audio-phrase .replay-button, .audio-ex .replay-button {
        width: 32px !important; height: 32px !important;
    }

    hr {
        border: 0; height: 1px; margin: 8px 0;
        background: linear-gradient(to right,
            rgba(0,0,0,0), rgba(0,0,0,0.18), rgba(0,0,0,0));
    }
    .meaning {
        font-size: 23px; font-weight: bold; color: #222;
        margin: 0 0 8px 0; text-align: left; line-height: 1.45;
    }
    .nightMode .meaning { color: #f0f0f0; }

    .example {
        background: #f7f9fa;
        padding: 10px 12px;
        border-left: 4px solid #0056b3;
        border-radius: 4px;
        color: #444; font-style: italic;
        font-size: 21px; line-height: 1.55;
        text-align: left;
        margin: 0 0 8px 0;
    }
    .nightMode .example { background: #252540; color: #d4d4d8; border-left-color: #7cb8ff; }

    .etymology {
        font-size: 19px; color: #555;
        background-color: #fffdf5;
        padding: 8px 12px; border-radius: 6px;
        margin: 6px 0 0 0; border: 1px solid #fef3c7;
        text-align: left; line-height: 1.45;
    }
    .nightMode .etymology { background-color: #1e2a1e; color: #a5d6a7; border-color: #2e4a2e; }
    .nightMode hr {
        background: linear-gradient(to right,
            rgba(255,255,255,0), rgba(255,255,255,0.12), rgba(255,255,255,0));
    }
    """

    DECK_ID = zlib.adler32(deck_name.encode('utf-8'))

    # Select model by card type; all use same 6 fields for compatibility
    model_id_map = {
        "standard": constants.ANKI_MODEL_ID,
        "cloze": constants.ANKI_MODEL_CLOZE_ID,
        "production": constants.ANKI_MODEL_PRODUCTION_ID,
        "translation": constants.ANKI_MODEL_TRANSLATION_ID,
    }
    model_id = model_id_map.get(card_type, constants.ANKI_MODEL_ID)

    # Template varies by card type: w=front, m/e/r=back (semantics differ per type)
    if card_type == "cloze":
        qfmt = '<div class="phrase" style="font-size:20px;">{{Phrase}}</div><span class="audio-phrase">{{Audio_Phrase}}</span>'
        afmt = '{{FrontSide}}<hr><div class="meaning">{{Meaning}}</div>{{#Etymology}}<div class="etymology">🌱 {{Etymology}}</div>{{/Etymology}}<span class="audio-ex">{{Audio_Example}}</span>'
    elif card_type == "production":
        qfmt = '<div class="phrase" style="font-size:22px;color:#333;">{{Phrase}}</div><span class="audio-phrase">{{Audio_Phrase}}</span>'
        afmt = '{{FrontSide}}<hr><div class="meaning">{{Meaning}}</div>{{#Example}}<div class="example">{{Example}}</div>{{/Example}}<span class="audio-ex">{{Audio_Example}}</span>'
    elif card_type == "translation":
        qfmt = '<div class="phrase" style="font-size:24px;">{{Phrase}}</div>'
        afmt = '{{FrontSide}}<hr><div class="meaning">{{Meaning}}</div>{{#Example}}<div class="example">{{Example}}</div>{{/Example}}<span class="audio-ex">{{Audio_Example}}</span>'
    else:
        qfmt = '<div class="phrase">{{Phrase}}</div><span class="audio-phrase">{{Audio_Phrase}}</span>'
        afmt = '{{FrontSide}}<hr><div class="meaning">{{Meaning}}</div>{{#Example}}<div class="example">{{Example}}</div>{{/Example}}<span class="audio-ex">{{Audio_Example}}</span>{{#Etymology}}<div class="etymology">🌱 {{Etymology}}</div>{{/Etymology}}'

    model = genanki.Model(
        model_id,
        f'VocabFlow {card_type.capitalize()}',
        fields=[
            {'name': 'Phrase'}, {'name': 'Meaning'},
            {'name': 'Example'}, {'name': 'Etymology'},
            {'name': 'Audio_Phrase'}, {'name': 'Audio_Example'}
        ],
        templates=[{'name': 'Card', 'qfmt': qfmt, 'afmt': afmt}],
        css=CSS
    )

    deck = genanki.Deck(DECK_ID, deck_name)

    # Pass 1: plan audio tasks using the persistent content-addressed cache.
    # Audio file paths are derived from hash(voice + text), so the same
    # word/sentence always maps to the same file.  tts.py skips files that
    # already exist, making retries cheap: only missing clips are re-fetched.
    if enable_tts:
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

    audio_tasks: List[Dict[str, str]] = []
    seen_audio_paths: set = set()   # dedup: avoid queuing the same file twice
    card_plans = []

    for card in cards_data:
        phrase = safe_str_clean(card.get('w', ''))
        meaning = safe_str_clean(card.get('m', ''))
        raw_example = safe_str_clean(card.get('e', ''))
        etymology = safe_str_clean(card.get('r', ''))
        note_id = card.get('id')

        example_sentences = _split_examples(raw_example)
        example_display = "<br>".join(
            f"• {s}" for s in example_sentences
        ) if example_sentences else ""

        phrase_plan = None      # (path, filename) or None
        example_plans = []      # [(path, filename), ...]

        if enable_tts and phrase:
            phrase_path = _audio_cache_path(phrase, tts_voice)
            phrase_filename = os.path.basename(phrase_path)
            if phrase_path not in seen_audio_paths:
                seen_audio_paths.add(phrase_path)
                audio_tasks.append({'text': phrase, 'path': phrase_path, 'voice': tts_voice})
            phrase_plan = (phrase_path, phrase_filename)

            if enable_example_tts:
                for sent in example_sentences:
                    tts_text = _example_text_for_tts(sent) if sent else ""
                    # Skip if too short — very short strings produce sub-threshold files
                    actual_text = tts_text if tts_text else sent
                    if not actual_text or len(actual_text) <= 3:
                        continue
                    ex_path = _audio_cache_path(actual_text, tts_voice)
                    ex_filename = os.path.basename(ex_path)
                    if ex_path not in seen_audio_paths:
                        seen_audio_paths.add(ex_path)
                        audio_tasks.append({'text': actual_text, 'path': ex_path, 'voice': tts_voice})
                    example_plans.append((ex_path, ex_filename))

        card_plans.append({
            'phrase': phrase,
            'meaning': meaning,
            'example_display': example_display,
            'etymology': etymology,
            'note_id': note_id,
            'phrase_plan': phrase_plan,
            'example_plans': example_plans,
        })

    # Run TTS — already-cached files are skipped by tts.py (os.path.exists check)
    if audio_tasks:
        def internal_progress(ratio: float, msg: str) -> None:
            if progress_callback:
                progress_callback(ratio, msg)

        run_async_batch(
            audio_tasks,
            concurrency=constants.TTS_CONCURRENCY,
            progress_callback=internal_progress,
        )

    # Pass 2: build notes; include [sound:] tags only for files that exist.
    # Count failures and collect failed phrase strings for retry UI.
    failed_audio_count = 0
    failed_phrases: List[str] = []
    for plan in card_plans:
        audio_phrase_field = ""
        audio_example_field = ""

        plan_has_failure = False
        if plan['phrase_plan']:
            phrase_path, phrase_filename = plan['phrase_plan']
            if os.path.exists(phrase_path):
                audio_phrase_field = f"[sound:{phrase_filename}]"
                media_files.append(phrase_path)
            else:
                failed_audio_count += 1
                plan_has_failure = True

        audio_example_parts = []
        for ex_path, ex_filename in plan['example_plans']:
            if os.path.exists(ex_path):
                audio_example_parts.append(f"[sound:{ex_filename}]")
                media_files.append(ex_path)
            else:
                failed_audio_count += 1
                plan_has_failure = True
        if plan_has_failure and plan['phrase']:
            failed_phrases.append(plan['phrase'])
        audio_example_field = "".join(audio_example_parts)

        fields = [
            plan['phrase'], plan['meaning'], plan['example_display'],
            plan['etymology'], audio_phrase_field, audio_example_field,
        ]

        note_id = plan['note_id']
        if note_id:
            note = genanki.Note(model=model, fields=fields, guid=note_id)
        else:
            note = genanki.Note(model=model, fields=fields)
        deck.add_note(note)

    if progress_callback:
        progress_callback(1.0, "📦 正在打包 .apkg 文件...")

    package = genanki.Package(deck)
    package.media_files = media_files   # only files confirmed to exist

    os.makedirs(APKG_TEMP_DIR, exist_ok=True)
    output_file = tempfile_mod.NamedTemporaryFile(
        dir=APKG_TEMP_DIR, delete=False, suffix='.apkg'
    )
    output_file.close()
    package.write_to_file(output_file.name)
    return output_file.name, failed_audio_count, failed_phrases

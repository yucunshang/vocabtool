# Anki package (.apkg) generation with optional TTS.

import os
import random
import tempfile
import time
import zlib
import re
from typing import Dict, List, Optional

import constants
from errors import ProgressCallback
from resources import get_genanki
from tts import run_async_batch
from utils import safe_str_clean

APKG_TEMP_DIR = os.path.join(tempfile.gettempdir(), constants.APKG_TEMP_SUBDIR)


def cleanup_old_apkg_files(max_age_seconds: int = constants.APKG_CLEANUP_MAX_AGE_SECONDS) -> None:
    """Remove .apkg files in our temp subdir older than max_age_seconds."""
    if not os.path.isdir(APKG_TEMP_DIR):
        return
    now = time.time()
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
    progress_callback: Optional[ProgressCallback] = None
) -> str:
    """Generate Anki package (.apkg) file with optional TTS audio.

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

    model = genanki.Model(
        constants.ANKI_MODEL_ID,
        'VocabFlow Unified Model',
        fields=[
            {'name': 'Phrase'}, {'name': 'Meaning'},
            {'name': 'Example'}, {'name': 'Etymology'},
            {'name': 'Audio_Phrase'}, {'name': 'Audio_Example'}
        ],
        templates=[{
            'name': 'Vocab Card',
            'qfmt': '<div class="phrase">{{Phrase}}</div>'
                     '<span class="audio-phrase">{{Audio_Phrase}}</span>',
            'afmt': '{{FrontSide}}'
                    '<hr>'
                    '<div class="meaning">{{Meaning}}</div>'
                    '{{#Example}}<div class="example">{{Example}}</div>{{/Example}}'
                    '<span class="audio-ex">{{Audio_Example}}</span>'
                    '{{#Etymology}}<div class="etymology">🌱 {{Etymology}}</div>{{/Etymology}}',
        }], css=CSS
    )

    deck = genanki.Deck(DECK_ID, deck_name)

    with tempfile_mod.TemporaryDirectory() as tmp_dir:
        notes_buffer = []
        audio_tasks = []

        for idx, card in enumerate(cards_data):
            phrase = safe_str_clean(card.get('w', ''))
            meaning = safe_str_clean(card.get('m', ''))
            raw_example = safe_str_clean(card.get('e', ''))
            etymology = safe_str_clean(card.get('r', ''))
            note_id = card.get('id')

            example_sentences = _split_examples(raw_example)
            example_display = "<br>".join(
                f"• {s}" for s in example_sentences
            ) if example_sentences else ""

            audio_phrase_field = ""
            audio_example_field = ""

            if enable_tts and phrase:
                safe_phrase = re.sub(r'[^a-zA-Z0-9]', '_', phrase)[:20]
                unique_id = int(time.time() * 1000) + random.randint(0, 9999)

                phrase_filename = f"tts_{safe_phrase}_{unique_id}_p.mp3"
                phrase_path = os.path.join(tmp_dir, phrase_filename)
                audio_tasks.append({
                    'text': phrase,
                    'path': phrase_path,
                    'voice': tts_voice
                })
                media_files.append(phrase_path)
                audio_phrase_field = f"[sound:{phrase_filename}]"

                audio_example_parts = []
                for ei, sent in enumerate(example_sentences):
                    if sent and len(sent) > 3:
                        ex_filename = f"tts_{safe_phrase}_{unique_id}_e{ei}.mp3"
                        ex_path = os.path.join(tmp_dir, ex_filename)
                        tts_text = _example_text_for_tts(sent)
                        audio_tasks.append({
                            'text': tts_text if tts_text else sent,
                            'path': ex_path,
                            'voice': tts_voice
                        })
                        media_files.append(ex_path)
                        audio_example_parts.append(f"[sound:{ex_filename}]")
                audio_example_field = "".join(audio_example_parts)

            fields = [
                phrase, meaning, example_display, etymology,
                audio_phrase_field, audio_example_field
            ]

            if note_id:
                note = genanki.Note(model=model, fields=fields, guid=note_id)
            else:
                note = genanki.Note(model=model, fields=fields)
            notes_buffer.append(note)

        if audio_tasks:
            def internal_progress(ratio: float, msg: str) -> None:
                if progress_callback:
                    progress_callback(ratio, msg)

            run_async_batch(audio_tasks, concurrency=constants.TTS_CONCURRENCY, progress_callback=internal_progress)

        for note in notes_buffer:
            deck.add_note(note)

        if progress_callback:
            progress_callback(1.0, "📦 正在打包 .apkg 文件...")

        package = genanki.Package(deck)
        package.media_files = [f for f in media_files if os.path.exists(f)]

        os.makedirs(APKG_TEMP_DIR, exist_ok=True)
        output_file = tempfile_mod.NamedTemporaryFile(
            dir=APKG_TEMP_DIR, delete=False, suffix='.apkg'
        )
        output_file.close()

        package.write_to_file(output_file.name)
        return output_file.name

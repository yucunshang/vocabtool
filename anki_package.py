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
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .meaning { font-size: 20px; font-weight: bold; color: #222; margin-bottom: 15px; text-align: left; }
    .nightMode .meaning { color: #e0e0e0; }
    .example {
        background: #f7f9fa;
        padding: 15px;
        border-left: 5px solid #0056b3;
        border-radius: 4px;
        color: #444;
        font-style: italic;
        font-size: 18px;
        line-height: 1.5;
        text-align: left;
        margin-bottom: 15px;
    }
    .nightMode .example { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .etymology { display: block; font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; border: 1px solid #fef3c7; }
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
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
            'qfmt': '''
                <div class="phrase">{{Phrase}}</div>
                <div>{{Audio_Phrase}}</div>
            ''',
            'afmt': '''
            {{FrontSide}}
            <hr>
            <div class="meaning">{{Meaning}}</div>
            {{#Example}}
            <div class="example">{{Example}}</div>
            {{/Example}}
            <div>{{Audio_Example}}</div>
            {{#Etymology}}
            <div class="etymology">🌱 词源: {{Etymology}}</div>
            {{/Etymology}}
            ''',
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
                        audio_tasks.append({
                            'text': sent,
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

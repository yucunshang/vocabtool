# Anki package (.apkg) generation with optional TTS.

import html
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

CARD_TEMPLATE_MODEL_OFFSETS = {
    "word_front": 1,
    "example_front": 2,
    "definition_front": 3,
}


def _get_card_template(card_template: str) -> str:
    if card_template in constants.CARD_TEMPLATES:
        return card_template
    return constants.DEFAULT_CARD_TEMPLATE


def _build_first_letter_hint(phrase: str) -> str:
    tokens = re.findall(r"[A-Za-z]+", phrase)
    if not tokens:
        return ""
    return " ".join(
        token[0].lower() + "_" * (len(token) - 1) if len(token) > 1 else token.lower()
        for token in tokens
    )


def _combine_meaning(chinese_meaning: str, english_definition: str) -> str:
    if chinese_meaning and english_definition:
        return f"{chinese_meaning}<br><span class=\"definition-inline\">{english_definition}</span>"
    return chinese_meaning or english_definition


def _highlight_target_in_example(example: str, phrase: str) -> str:
    if not example:
        return html.escape(phrase)
    if not phrase:
        return html.escape(example)

    pattern = re.compile(
        rf"(?<![A-Za-z])({re.escape(phrase)})(?![A-Za-z])",
        flags=re.IGNORECASE,
    )
    match = pattern.search(example)
    if not match:
        return html.escape(example)

    return (
        html.escape(example[:match.start()])
        + f"<strong>{html.escape(match.group(0))}</strong>"
        + html.escape(example[match.end():])
    )


def _get_anki_template(card_template: str) -> Dict[str, str]:
    templates = {
        "word_front": {
            "name": "Word Front",
            "qfmt": '''
                <div class="front-word">{{Phrase}}</div>
                <div class="audio">{{Audio_Phrase}}</div>
            ''',
            "afmt": '''
                {{FrontSide}}
                <hr>
                <div class="meaning">{{ChineseMeaning}}</div>
                {{#EnglishDefinition}}
                <div class="definition">{{EnglishDefinition}}</div>
                {{/EnglishDefinition}}
                {{#EnglishExample}}
                <div class="example">{{EnglishExample}}</div>
                {{/EnglishExample}}
                {{#ChineseExample}}
                <div class="example-cn">{{ChineseExample}}</div>
                {{/ChineseExample}}
                <div class="audio">{{Audio_Example}}</div>
            ''',
        },
        "example_front": {
            "name": "Example Front",
            "qfmt": '''
                <div class="front-example">{{ExampleFront}}</div>
                <div class="audio">{{Audio_Example}}</div>
            ''',
            "afmt": '''
                {{FrontSide}}
                <hr>
                <div class="answer-word">{{Phrase}}</div>
                <div class="meaning">{{ChineseMeaning}}</div>
                {{#EnglishDefinition}}
                <div class="definition">{{EnglishDefinition}}</div>
                {{/EnglishDefinition}}
                {{#EnglishExample}}
                <div class="example">{{EnglishExample}}</div>
                {{/EnglishExample}}
                {{#ChineseExample}}
                <div class="example-cn">{{ChineseExample}}</div>
                {{/ChineseExample}}
            ''',
        },
        "definition_front": {
            "name": "Definition Front",
            "qfmt": '''
                <div class="front-definition">{{EnglishDefinition}}</div>
                {{#PartOfSpeech}}<div class="meta">{{PartOfSpeech}}</div>{{/PartOfSpeech}}
                {{#Hint}}<div class="hint">Hint: {{Hint}}</div>{{/Hint}}
            ''',
            "afmt": '''
                {{FrontSide}}
                <hr>
                <div class="answer-word">{{Phrase}}</div>
                <div class="meaning">{{ChineseMeaning}}</div>
                {{#EnglishExample}}
                <div class="example">{{EnglishExample}}</div>
                {{/EnglishExample}}
                {{#ChineseExample}}
                <div class="example-cn">{{ChineseExample}}</div>
                {{/ChineseExample}}
                <div class="audio">{{Audio_Phrase}}</div>
            ''',
        },
    }
    return templates[_get_card_template(card_template)]


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


def generate_anki_package(
    cards_data: List[Dict[str, str]],
    deck_name: str,
    enable_tts: bool = False,
    tts_voice: str = "en-US-JennyNeural",
    progress_callback: Optional[ProgressCallback] = None,
    card_template: str = constants.DEFAULT_CARD_TEMPLATE,
) -> str:
    """Generate Anki package (.apkg) file with optional TTS audio."""
    genanki, tempfile_mod = get_genanki()
    media_files = []
    card_template = _get_card_template(card_template)
    selected_template = _get_anki_template(card_template)

    CSS = """
    .card { font-family: Arial, sans-serif; font-size: 20px; text-align: center; color: #243041; background-color: #fff; padding: 22px; line-height: 1.55; }
    hr { border: 0; height: 1px; background: #d7dde7; margin: 18px 0; }
    .front-word, .answer-word { font-size: 34px; font-weight: 800; color: #105f7a; margin-bottom: 10px; }
    .front-example { font-size: 25px; line-height: 1.55; text-align: left; color: #243041; }
    .front-example strong { color: #0f766e; font-weight: 800; }
    .front-definition { font-size: 25px; line-height: 1.45; color: #243041; margin-bottom: 12px; }
    .meta { display: inline-block; font-size: 15px; color: #526071; background: #eef6f8; border: 1px solid #cfe4ea; border-radius: 999px; padding: 3px 10px; margin: 4px 0 10px; }
    .hint { display: inline-block; font-size: 18px; letter-spacing: 1px; color: #0f766e; background: #eefbf7; border: 1px solid #b7ead8; border-radius: 8px; padding: 6px 12px; margin-top: 8px; }
    .meaning { font-size: 21px; font-weight: 700; color: #1d2733; margin-bottom: 12px; text-align: left; }
    .definition { font-size: 19px; color: #435060; margin-bottom: 14px; text-align: left; }
    .definition-inline { color: #435060; font-weight: 500; }
    .example { background: #f6f8fa; padding: 14px; border-left: 5px solid #0f766e; border-radius: 6px; color: #2e3b4a; font-size: 22px; line-height: 1.5; text-align: left; margin-bottom: 10px; }
    .example-cn { color: #526071; font-size: 18px; text-align: left; margin-bottom: 12px; }
    .audio { margin-top: 8px; }
    .nightMode .card { color: #e5e7eb; background-color: #1f2937; }
    .nightMode .front-word, .nightMode .answer-word, .nightMode .front-definition, .nightMode .front-example { color: #e5e7eb; }
    .nightMode .meaning { color: #f3f4f6; }
    .nightMode .definition, .nightMode .example-cn, .nightMode .definition-inline { color: #cbd5e1; }
    .nightMode .example { background: #303846; color: #e5e7eb; border-left-color: #5eead4; }
    .nightMode .meta { background: #263241; color: #cbd5e1; border-color: #3f4f63; }
    .nightMode .hint { background: #12312f; color: #99f6e4; border-color: #1f5f58; }
    """

    DECK_ID = zlib.adler32(deck_name.encode('utf-8'))
    model_id = constants.ANKI_MODEL_ID_BASE + CARD_TEMPLATE_MODEL_OFFSETS[card_template]
    model_label = constants.CARD_TEMPLATES[card_template]["label"]

    model = genanki.Model(
        model_id,
        f'VocabFlow {model_label}',
        fields=[
            {'name': 'Phrase'}, {'name': 'PartOfSpeech'},
            {'name': 'ChineseMeaning'}, {'name': 'EnglishDefinition'},
            {'name': 'Meaning'}, {'name': 'EnglishExample'},
            {'name': 'ChineseExample'}, {'name': 'Hint'},
            {'name': 'ExampleFront'}, {'name': 'Etymology'},
            {'name': 'Audio_Phrase'}, {'name': 'Audio_Example'}
        ],
        templates=[selected_template],
        css=CSS
    )

    deck = genanki.Deck(DECK_ID, deck_name)

    with tempfile_mod.TemporaryDirectory() as tmp_dir:
        notes_buffer = []
        audio_tasks = []

        for idx, card in enumerate(cards_data):
            phrase = safe_str_clean(card.get('w', ''))
            pos = safe_str_clean(card.get('pos', ''))
            chinese_meaning = safe_str_clean(card.get('cn', ''))
            english_definition = safe_str_clean(card.get('en', ''))
            meaning = safe_str_clean(card.get('m', ''))
            if not chinese_meaning:
                chinese_meaning = meaning
            if not meaning:
                meaning = _combine_meaning(chinese_meaning, english_definition)
            example = safe_str_clean(card.get('e', ''))
            chinese_example = safe_str_clean(card.get('ec', ''))
            hint = safe_str_clean(card.get('hint', '')) or _build_first_letter_hint(phrase)
            example_front = _highlight_target_in_example(example, phrase)
            etymology = safe_str_clean(card.get('r', ''))
            note_id = card.get('id')

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

                if example and len(example) > 3:
                    example_filename = f"tts_{safe_phrase}_{unique_id}_e.mp3"
                    example_path = os.path.join(tmp_dir, example_filename)
                    audio_tasks.append({
                        'text': example,
                        'path': example_path,
                        'voice': tts_voice
                    })
                    media_files.append(example_path)
                    audio_example_field = f"[sound:{example_filename}]"

            if note_id:
                note = genanki.Note(
                    model=model,
                    fields=[
                        phrase, pos, chinese_meaning, english_definition, meaning,
                        example, chinese_example, hint, example_front, etymology,
                        audio_phrase_field, audio_example_field
                    ],
                    guid=note_id
                )
            else:
                note = genanki.Note(
                    model=model,
                    fields=[
                        phrase, pos, chinese_meaning, english_definition, meaning,
                        example, chinese_example, hint, example_front, etymology,
                        audio_phrase_field, audio_example_field
                    ]
                )
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

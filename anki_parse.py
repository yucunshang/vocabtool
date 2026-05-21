# Parse AI-generated text into structured Anki card data.

import re
from typing import Dict, List


def build_first_letter_hint(phrase: str) -> str:
    """Build a first-letter hint such as `h_____` or `t____ w___`."""
    tokens = re.findall(r"[A-Za-z]+", phrase)
    if not tokens:
        return ""
    hinted_tokens = []
    for token in tokens:
        if len(token) <= 1:
            hinted_tokens.append(token.lower())
        else:
            hinted_tokens.append(token[0].lower() + "_" * (len(token) - 1))
    return " ".join(hinted_tokens)


def _combine_meaning(chinese_meaning: str, english_definition: str) -> str:
    if chinese_meaning and english_definition:
        return f"{chinese_meaning}<br><span class=\"definition-inline\">{english_definition}</span>"
    return chinese_meaning or english_definition


def parse_anki_data(raw_text: str) -> List[Dict[str, str]]:
    """Parse AI-generated text into structured Anki card data."""
    parsed_cards = []
    text = raw_text.strip()

    code_blocks = re.findall(r'```(?:text|csv)?\s*(.*?)\s*```', text, re.DOTALL)

    if code_blocks:
        text = "\n".join(code_blocks)
    else:
        text = re.sub(r'^```.*$', '', text, flags=re.MULTILINE)

    lines = text.split('\n')
    seen_phrases = set()

    for line in lines:
        line = line.strip()
        if not line or "|||" not in line:
            continue

        parts = line.split("|||")
        if len(parts) < 2:
            continue

        phrase = parts[0].strip()

        if len(parts) >= 6:
            pos = parts[1].strip()
            chinese_meaning = parts[2].strip()
            english_definition = parts[3].strip()
            example = parts[4].strip()
            chinese_example = parts[5].strip()
            meaning = _combine_meaning(chinese_meaning, english_definition)
            etymology = ""
        else:
            pos = ""
            meaning = parts[1].strip()
            chinese_meaning = meaning
            english_definition = ""
            example = parts[2].strip() if len(parts) > 2 else ""
            chinese_example = ""
            etymology = parts[3].strip() if len(parts) > 3 else ""

        if not phrase or not meaning:
            continue

        if phrase.lower() in seen_phrases:
            continue
        seen_phrases.add(phrase.lower())

        parsed_cards.append({
            'w': phrase,
            'pos': pos,
            'cn': chinese_meaning,
            'en': english_definition,
            'm': meaning,
            'e': example,
            'ec': chinese_example,
            'hint': build_first_letter_hint(phrase),
            'r': etymology
        })

    return parsed_cards

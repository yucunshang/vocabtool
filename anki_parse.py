# Parse AI-generated text into structured Anki card data.

import re
from typing import Dict, List, Optional


def _is_cloze_line(phrase: str) -> bool:
    """Detect cloze blank: 8 underscores OR first-letter + 7 underscores (e.g. b_______)."""
    return bool(re.search(r'_{7,}|[A-Za-z]_{7}', phrase or ""))


def parse_anki_data(raw_text: str, expected_card_type: Optional[str] = None) -> List[Dict[str, str]]:
    """Parse AI-generated text into structured Anki card data.

    Supports variable field counts (2–4 fields separated by ``|||``):
      Standard cards (3-4 fields):
        Field 1: word / phrase  → key ``w``
        Field 2: definition     → key ``m``
        Field 3: example(s)     → key ``e``
        Field 4: etymology      → key ``r``
      Cloze cards (3 fields):
        Field 1: sentence with blank → key ``w``
        Field 2: word /IPA/ meaning  → key ``m``
        Field 3: full sentence + CN  → key ``e``
    """
    parsed_cards: List[Dict[str, str]] = []
    text = raw_text.strip()

    code_blocks = re.findall(r'```(?:text|csv)?\s*(.*?)\s*```', text, re.DOTALL)

    if code_blocks:
        text = "\n".join(code_blocks)
    else:
        text = re.sub(r'^```.*$', '', text, flags=re.MULTILINE)

    lines = text.split('\n')
    seen_phrases: set = set()

    for line in lines:
        line = line.strip()
        if not line or "|||" not in line:
            continue

        parts = [p.strip() for p in line.split("|||")]
        if len(parts) < 2:
            continue

        phrase = parts[0]
        meaning = parts[1]

        if not phrase or not meaning:
            continue

        is_cloze = _is_cloze_line(phrase) or expected_card_type == "cloze"

        if is_cloze:
            example = parts[2] if len(parts) > 2 else ""
            card = {'w': phrase, 'm': meaning, 'e': example, 'r': '', 'ct': 'cloze'}
            dedup_key = phrase.lower()
        else:
            example = parts[2] if len(parts) > 2 else ""
            etymology = parts[3] if len(parts) > 3 else ""
            card = {'w': phrase, 'm': meaning, 'e': example, 'r': etymology, 'ct': 'standard'}
            dedup_key = phrase.lower()

        if dedup_key in seen_phrases:
            continue
        seen_phrases.add(dedup_key)
        parsed_cards.append(card)

    return parsed_cards

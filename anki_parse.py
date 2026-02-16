# Parse AI-generated text into structured Anki card data.

import re
from typing import Dict, List


def parse_anki_data(raw_text: str) -> List[Dict[str, str]]:
    """Parse AI-generated text into structured Anki card data.

    Supports variable field counts (2–4 fields separated by ``|||``):
      Field 1: word / phrase  (always required → key ``w``)
      Field 2: definition     (always required → key ``m``)
      Field 3: example(s)     (optional → key ``e``; may contain ``//`` separators)
      Field 4: etymology      (optional → key ``r``)
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

        parts = line.split("|||")
        if len(parts) < 2:
            continue

        phrase = parts[0].strip()
        meaning = parts[1].strip()
        example = parts[2].strip() if len(parts) > 2 else ""
        etymology = parts[3].strip() if len(parts) > 3 else ""

        if not phrase or not meaning:
            continue

        if phrase.lower() in seen_phrases:
            continue
        seen_phrases.add(phrase.lower())

        parsed_cards.append({
            'w': phrase,
            'm': meaning,
            'e': example,
            'r': etymology,
        })

    return parsed_cards

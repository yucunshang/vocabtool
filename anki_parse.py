# Parse AI-generated text into structured Anki card data.

import re
from typing import Dict, List, Optional


def parse_anki_data(raw_text: str) -> List[Dict[str, str]]:
    """Parse AI-generated text into structured Anki card data.

    Supports variable field counts:
      - 4 fields: phrase ||| meaning ||| example ||| etymology (standard/production/translation)
      - 5 fields: cloze ||| word/phonetic ||| 释义 ||| collocations ||| example (cloze card)
    """
    parsed_cards: List[Dict[str, str]] = []
    text = raw_text.strip()

    code_blocks = re.findall(r'```(?:text|csv)?\s*(.*?)\s*```', text, re.DOTALL)

    if code_blocks:
        text = "\n".join(code_blocks)
    else:
        text = re.sub(r'^```.*$', '', text, flags=re.MULTILINE)

    seen_phrases: set = set()

    def _parse_parts(parts: list) -> Optional[Dict[str, str]]:
        if len(parts) >= 4 and "________" in (parts[0] or ""):
            phrase = parts[0].strip()
            meaning = "\n".join(p.strip() for p in parts[1:3] if p.strip())
            example = parts[3].strip() if len(parts) > 3 else ""
            return {'w': phrase, 'm': meaning, 'e': example, 'r': ""}
        if len(parts) >= 2:
            phrase = parts[0].strip()
            meaning = parts[1].strip()
            example = parts[2].strip() if len(parts) > 2 else ""
            etymology = parts[3].strip() if len(parts) > 3 else ""
            return {'w': phrase, 'm': meaning, 'e': example, 'r': etymology}
        return None

    # Block format: cards separated by blank line(s)
    if "\n\n" in text or re.search(r'\n\s{2,}\n', text):
        blocks = re.split(r'\n\s*\n', text)
        block_cards: List[Dict[str, str]] = []
        for block in blocks:
            block = block.strip()
            if not block or "|||" not in block:
                continue
            parts = [p.strip() for p in block.split("|||")]
            card = _parse_parts(parts)
            if not card or not card["w"] or not card["m"]:
                continue
            if card["w"].lower() in seen_phrases:
                continue
            seen_phrases.add(card["w"].lower())
            block_cards.append(card)
        if block_cards:
            return block_cards
        seen_phrases.clear()

    # Line format: one card per line
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or "|||" not in line:
            continue
        parts = [p.strip() for p in line.split("|||")]
        card = _parse_parts(parts)
        if not card or not card["w"] or not card["m"]:
            continue
        if card["w"].lower() in seen_phrases:
            continue
        seen_phrases.add(card["w"].lower())
        parsed_cards.append(card)

    return parsed_cards

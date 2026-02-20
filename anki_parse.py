# Parse AI-generated text into structured Anki card data.

import re
from typing import Dict, List, Optional


def parse_anki_data(raw_text: str) -> List[Dict[str, str]]:
    """Parse AI-generated text into structured Anki card data.

    Supports variable field counts:
      - 3 fields reading: phrase(________) ||| meaning ||| example
      - 4 fields: phrase ||| meaning ||| example ||| etymology (standard/production/translation)
    """
    parsed_cards: List[Dict[str, str]] = []
    text = raw_text.strip()

    code_blocks = re.findall(r'```(?:text|csv)?\s*(.*?)\s*```', text, re.DOTALL)

    if code_blocks:
        text = "\n".join(code_blocks)
    else:
        text = re.sub(r'^```.*$', '', text, flags=re.MULTILINE)

    seen_phrases: set = set()

    def _is_cloze_phrase(s: str) -> bool:
        return "________" in (s or "") or "{{c1::" in (s or "")

    def _parse_parts(parts: list) -> Optional[Dict[str, str]]:
        # Reading card: 3 fields (phrase with ________ ||| meaning ||| example) or legacy 4–5 fields
        if len(parts) >= 3 and _is_cloze_phrase(parts[0]):
            phrase = parts[0].strip()
            meaning = parts[1].strip()
            example = parts[2].strip() if len(parts) > 2 else ""
            return {'w': phrase, 'm': meaning, 'e': example, 'r': ""}
        if len(parts) >= 2:
            phrase = parts[0].strip()
            meaning = parts[1].strip()
            example = parts[2].strip() if len(parts) > 2 else ""
            etymology = parts[3].strip() if len(parts) > 3 else ""
            return {'w': phrase, 'm': meaning, 'e': example, 'r': etymology}
        return None

    # Block format: blank lines separate blocks; each block can have multiple cards (one per line)
    if "\n\n" in text or re.search(r'\n\s{2,}\n', text):
        blocks = re.split(r'\n\s*\n', text)
        block_cards: List[Dict[str, str]] = []
        for block in blocks:
            for line in block.split('\n'):
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

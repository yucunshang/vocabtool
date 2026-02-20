# Parse AI-generated text into structured Anki card data.

import re
from typing import Dict, List, Optional

_RELAXED_DELIMITER_RE = re.compile(r"\s*\|\s*\|\s*\|\s*")


def _extract_candidate_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    code_blocks = re.findall(r"```(?:text|csv)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_blocks:
        text = "\n".join(code_blocks)
    else:
        text = re.sub(r"^```.*$", "", text, flags=re.MULTILINE)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _is_cloze_phrase(s: str) -> bool:
    return "________" in (s or "") or "{{c1::" in (s or "")


def _normalize_line_for_split(line: str) -> str:
    # Normalize full-width bars and relaxed spaced variants, then trim parts.
    norm = (line or "").replace("｜", "|").strip()
    if not norm:
        return ""
    norm = _RELAXED_DELIMITER_RE.sub("|||", norm)
    if "|||" not in norm:
        return norm
    parts = [p.strip() for p in norm.split("|||")]
    while len(parts) > 1 and not parts[-1]:
        parts.pop()
    return " ||| ".join(parts).strip()


def _repair_parts(parts: List[str], is_cloze: bool) -> List[str]:
    repaired = [p.strip() for p in parts]
    while len(repaired) > 1 and not repaired[-1]:
        repaired.pop()

    # Common malformed outputs: one extra delimiter splitting example/etymology.
    if is_cloze:
        if len(repaired) == 4:
            # Expected 3 fields; fold overflow back into example field.
            return [repaired[0], repaired[1], " ||| ".join(repaired[2:]).strip()]
        if len(repaired) > 5:
            return repaired[:4] + [" ||| ".join(repaired[4:]).strip()]
        return repaired

    if len(repaired) > 4:
        return repaired[:3] + [" ||| ".join(repaired[3:]).strip()]
    return repaired


def _parse_parts(parts: List[str]) -> Optional[Dict[str, str]]:
    if not parts:
        return None
    is_cloze = _is_cloze_phrase(parts[0])
    parts = _repair_parts(parts, is_cloze)

    # Reading card: 3 fields (phrase ||| meaning ||| example)
    # or legacy 5 fields (phrase ||| word/IPA ||| 释义 ||| 搭配 ||| example)
    if len(parts) >= 5 and is_cloze:
        phrase = parts[0].strip()
        meaning = "\n".join(p.strip() for p in parts[1:4] if p.strip())
        example = parts[4].strip()
        return {"w": phrase, "m": meaning, "e": example, "r": "", "ct": "cloze"}
    if len(parts) >= 3 and is_cloze:
        phrase = parts[0].strip()
        meaning = parts[1].strip()
        example = parts[2].strip()
        return {"w": phrase, "m": meaning, "e": example, "r": "", "ct": "cloze"}
    if len(parts) >= 2:
        phrase = parts[0].strip()
        meaning = parts[1].strip()
        example = parts[2].strip() if len(parts) > 2 else ""
        etymology = parts[3].strip() if len(parts) > 3 else ""
        return {"w": phrase, "m": meaning, "e": example, "r": etymology, "ct": "standard"}
    return None


def _dedupe_key(card: Dict[str, str]) -> tuple:
    """Build a dedupe key that avoids dropping translation cards with same CN gloss."""
    ct = (card.get("ct") or "standard").strip().lower()
    w = (card.get("w") or "").strip().lower()
    m = (card.get("m") or "").strip().lower()
    if ct == "cloze":
        return (ct, w)
    return (ct, w, m)


def parse_anki_data(raw_text: str) -> List[Dict[str, str]]:
    """Parse AI-generated text into structured Anki card data.

    Supports variable field counts:
      - 3 fields reading: phrase(________) ||| meaning ||| example
      - 4 fields: phrase ||| meaning ||| example ||| etymology (standard/production/translation)
    """
    text = _extract_candidate_text(raw_text)
    if not text:
        return []

    parsed_cards: List[Dict[str, str]] = []
    seen_phrases: set = set()

    for raw_line in text.split("\n"):
        line = _normalize_line_for_split(raw_line)
        if not line or "|||" not in line:
            continue
        parts = [p.strip() for p in line.split("|||")]
        card = _parse_parts(parts)
        if not card or not card["w"] or not card["m"]:
            continue
        key = _dedupe_key(card)
        if key in seen_phrases:
            continue
        seen_phrases.add(key)
        parsed_cards.append(card)

    return parsed_cards

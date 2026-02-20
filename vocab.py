# Word validation and text analysis (rank-based vocabulary extraction).

import functools
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import constants
from resources import get_vocab_dict, load_nlp_resources

# Common words that should always be treated as basic vocabulary.
# Months and weekdays often missing from NGSL or ranked oddly high.
_BASIC_WORDS: Dict[str, int] = {
    "january": 400, "february": 400, "march": 400, "april": 400,
    "may": 400, "june": 400, "july": 400, "august": 400,
    "september": 400, "october": 400, "november": 400, "december": 400,
    "monday": 500, "tuesday": 500, "wednesday": 500, "thursday": 500,
    "friday": 500, "saturday": 500, "sunday": 500,
}


@functools.lru_cache(maxsize=None)
def _vocab_dict() -> Dict[str, int]:
    """Return a copy of VOCAB_DICT with basic-word overrides applied."""
    d = dict(get_vocab_dict())  # copy — never mutate the cached original
    for w, r in _BASIC_WORDS.items():
        if w not in d or d[w] > r:
            d[w] = r
    return d


def is_valid_word(word: str) -> bool:
    """Validate if a word meets criteria for processing."""
    if len(word) < constants.MIN_WORD_LENGTH or len(word) > constants.MAX_WORD_LENGTH:
        return False
    if re.search(r'(.)\1{2,}', word):
        return False
    if not re.search(r'[aeiouy]', word):
        return False
    return True


def get_lemma(word: str, lemminflect: Any) -> str:
    """Get lemma of a word. Tries NOUN, VERB, ADJ, ADV; returns first non-empty result."""
    if not word:
        return word
    for upos in ("NOUN", "VERB", "ADJ", "ADV"):
        try:
            lemmas = lemminflect.getLemma(word, upos=upos)
            if lemmas:
                return lemmas[0]
        except Exception:
            continue
    return word


def get_lemma_for_word(word: str) -> str:
    """Public helper: load NLP resources and return lemma for a single word."""
    _, lemminflect = load_nlp_resources()
    return get_lemma(word.strip().lower(), lemminflect)


def analyze_logic(
    text: str,
    current_level: int,
    target_level: int,
    include_unknown: bool
) -> Tuple[List[Tuple[str, int]], int, Dict[str, float]]:
    """Analyze text to extract vocabulary within specified rank range."""
    nltk, lemminflect = load_nlp_resources()
    vocab_dict = _vocab_dict()

    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_raw_count = len(raw_tokens)

    valid_tokens = [token.lower() for token in raw_tokens if is_valid_word(token.lower())]
    token_counts = Counter(valid_tokens)

    stats_known_count = 0
    stats_target_count = 0
    stats_valid_total = sum(token_counts.values())

    final_candidates = []
    seen_lemmas = set()

    for word, count in token_counts.items():
        lemma = get_lemma(word, lemminflect)
        # Try exact case first so "May" vs "may" are distinct
        rank_lemma = vocab_dict.get(lemma, vocab_dict.get(lemma.lower(), 99999)) if lemma else 99999
        rank_orig = vocab_dict.get(word, 99999)  # word is already lowercase

        best_rank = min(rank_lemma, rank_orig)

        if best_rank < current_level:
            stats_known_count += count
            include_word = False
        elif current_level <= best_rank <= target_level:
            stats_target_count += count
            include_word = True
        else:
            include_word = (best_rank == 99999 and include_unknown)

        if include_word:
            word_to_keep = lemma
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)

    final_candidates.sort(key=lambda x: x[1])

    coverage_ratio = (stats_known_count / stats_valid_total) if stats_valid_total > 0 else 0
    target_ratio = (stats_target_count / stats_valid_total) if stats_valid_total > 0 else 0

    stats_info = {
        "coverage": coverage_ratio,
        "target_density": target_ratio
    }

    return final_candidates, total_raw_count, stats_info

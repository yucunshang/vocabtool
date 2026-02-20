# Cached resource loaders (NLP, file parsers, genanki, vocab data).

import functools
import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_vocab_dict() -> Dict[str, int]:
    """Return the vocabulary dict from the cached load_vocab_data() call."""
    return load_vocab_data()[0]


def get_rank_for_word(word: str) -> int:
    """Get rank for word: try exact case first, then lowercase. Returns 99999 if not found."""
    if not word:
        return 99999
    vocab = load_vocab_data()[0]
    w = word.strip()
    return vocab.get(w, vocab.get(w.lower(), 99999))


@functools.lru_cache(maxsize=None)
def load_nlp_resources() -> Tuple[Any, Any]:
    """Load NLTK and lemminflect resources with proper error handling."""
    import nltk
    import lemminflect

    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

        required_packages = [
            ('taggers', 'averaged_perceptron_tagger'),
            ('tokenizers', 'punkt'),
            ('tokenizers', 'punkt_tab'),
        ]
        for category, pkg in required_packages:
            try:
                nltk.data.find(f'{category}/{pkg}')
            except LookupError:
                logger.info("Downloading NLTK package: %s", pkg)
                nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)  # pkg name only, no category
    except Exception as e:
        logger.error("NLP 资源加载失败: %s", e)

    return nltk, lemminflect


@functools.lru_cache(maxsize=None)
def get_file_parsers() -> Tuple[Any, Any, Any, Any, Any]:
    """Lazy load file parsing libraries (cached)."""
    import pypdf
    import docx
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    return pypdf, docx, ebooklib, epub, BeautifulSoup


@functools.lru_cache(maxsize=None)
def get_genanki() -> Tuple[Any, Any]:
    """Lazy load genanki library (cached)."""
    import genanki
    import tempfile
    return genanki, tempfile


@functools.lru_cache(maxsize=None)
def load_vocab_data() -> Tuple[Dict[str, int], Optional[pd.DataFrame]]:
    """Load vocabulary data from CSV files only."""
    root_dir = os.path.dirname(os.path.abspath(__file__))

    possible_names = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next(
        (os.path.join(root_dir, f) for f in possible_names
         if os.path.exists(os.path.join(root_dir, f))),
        None
    )

    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]

            if len(df.columns) < 2:
                logger.error("CSV file %s has fewer than 2 columns", file_path)
                return {}, None
            word_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            rank_col = next((c for c in df.columns if 'rank' in c), df.columns[1])

            df = df.dropna(subset=[word_col])
            # Preserve case so "May" (month) and "may" (modal) can both exist
            df[word_col] = df[word_col].astype(str).str.strip()
            df[rank_col] = pd.to_numeric(df[rank_col], errors='coerce')
            df = df.sort_values(rank_col).drop_duplicates(subset=[word_col], keep='first')

            vocab_dict = pd.Series(df[rank_col].values, index=df[word_col]).to_dict()
            return vocab_dict, df
        except Exception as e:
            logger.error("Error loading CSV file %s: %s", file_path, e)
            return {}, None

    logger.warning("No vocabulary data files found (searched in %s)", root_dir)
    return {}, None

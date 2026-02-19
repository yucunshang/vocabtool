# AI-backed word definitions and batch card generation.
# 提示词模板见 prompts.py，可直接修改，不影响制卡 / apkg / 语音 / 卡片格式。

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import constants
from config import get_config
from errors import ErrorHandler
from prompts import CARD_GEN_SYSTEM_PROMPT, CARD_GEN_USER_TEMPLATE, LOOKUP_SYSTEM_PROMPT
from resources import get_rank_for_word

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("OpenAI library not available")


class CardFormat(TypedDict, total=False):
    front: str        # "word" | "phrase"
    definition: str   # "cn" | "en" | "en_native" | "both"
    examples: int     # 1 | 2 | 3
    etymology: bool   # True | False
    examples_with_cn: bool   # True = 例句带中文翻译
    examples_colloquial: bool   # True = 例句用口语


# Default: 正面单词，反面中文释义，2条例句带翻译，不要词源
DEFAULT_CARD_FORMAT: CardFormat = {
    "front": "word",
    "definition": "cn",
    "examples": 2,
    "etymology": False,
    "examples_with_cn": True,
}

# Fast in-memory cache for quick lookup to match vocabtool behavior.
_QUERY_CACHE: OrderedDict[str, str] = OrderedDict()
_QUERY_CACHE_MAX = 500
_OPENAI_CLIENT: Optional[Any] = None


def build_card_prompt(words_str: str, fmt: Optional[CardFormat] = None) -> str:
    """Build the built-in AI card generation prompt. Template is fixed (Lexicographer/Etymologist); fmt is ignored but kept for API compatibility."""
    return CARD_GEN_USER_TEMPLATE.format(words_str=words_str)


def get_openai_client() -> Optional[Any]:
    """Get configured OpenAI client with proper error handling."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    if not OpenAI:
        logger.error("OpenAI library not installed")
        return None

    cfg = get_config()
    api_key = cfg["openai_api_key"]
    if not api_key:
        logger.error("OPENAI_API_KEY not configured in secrets or env")
        return None

    try:
        _OPENAI_CLIENT = OpenAI(
            api_key=api_key,
            base_url=cfg["openai_base_url"],
            timeout=30.0,
        )
        return _OPENAI_CLIENT
    except Exception as e:
        ErrorHandler.handle(e, "Failed to initialize OpenAI client")
        return None


def _rank_from_ai_content(content: str, fallback_rank: int) -> int:
    """Return the rank of the headword the AI chose to define.

    The AI normalises inflected forms, misspellings, and non-standard casing
    to the dictionary base form on its first content line:
        'headword (pos CN-pos)'
    Extracting that headword gives a correct rank for inputs like 'cats',
    'RUNNING', 'ran', or 'recieve'.  The spell-correction notice line
    (✏️ 拼写纠正: ...) is skipped when present.
    Falls back to fallback_rank if no headword can be parsed.
    """
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('✏️ 拼写纠正:'):
            continue
        if '(' in line:
            headword = line.split('(', 1)[0].strip()
            if headword:
                return get_rank_for_word(headword)
        break
    return fallback_rank


def get_word_quick_definition(
    word: str,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Get ultra-concise word definition using AI, with rank info.

    If ``stream_callback`` is provided, tokens are streamed incrementally.
    """
    word_clean = word.strip()
    rank = get_rank_for_word(word_clean)

    client = get_openai_client()
    if not client:
        return {"error": "AI client not available"}

    model_name = get_config()["openai_model"]

    # Case-sensitive cache: "China" vs "china", "May" vs "may" get different results
    cache_key = word_clean
    if cache_key in _QUERY_CACHE:
        _QUERY_CACHE.move_to_end(cache_key)
        cached = _QUERY_CACHE[cache_key]
        return {"result": cached, "rank": _rank_from_ai_content(cached, rank), "cached": True}

    try:
        if stream_callback is not None:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": LOOKUP_SYSTEM_PROMPT},
                    {"role": "user", "content": word_clean}
                ],
                temperature=0.3,
                max_tokens=300,
                stream=True,
            )
            chunks: List[str] = []
            for event in stream:
                delta = ""
                if getattr(event, "choices", None):
                    delta = (event.choices[0].delta.content or "")
                if delta:
                    chunks.append(delta)
                    stream_callback("".join(chunks))
            content = "".join(chunks).strip()
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": LOOKUP_SYSTEM_PROMPT},
                    {"role": "user", "content": word_clean}
                ],
                temperature=0.3,
                max_tokens=300,
            )
            content = (response.choices[0].message.content or "").strip()
        if not content:
            return {"error": "AI 返回为空"}

        _QUERY_CACHE[cache_key] = content
        if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
            _QUERY_CACHE.popitem(last=False)

        return {"result": content, "rank": _rank_from_ai_content(content, rank)}

    except Exception as e:
        logger.error("Error getting definition: %s", e)
        return {"error": str(e)}


def _process_one_batch(
    batch_index: int,
    batch: List[str],
    client: Any,
    model_name: str,
    card_format: Optional[CardFormat],
) -> Tuple[int, str]:
    """Process a single batch, return (batch_index, content). Empty string on failure."""
    current_batch_str = ", ".join(batch)
    user_prompt = build_card_prompt(current_batch_str, card_format)
    max_attempts = constants.AI_BATCH_MAX_RETRIES
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": CARD_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=4096,  # 限制输出，避免超长响应浪费 tokens
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("Empty AI batch response")
            return (batch_index, content)
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            else:
                ErrorHandler.handle(
                    e,
                    f"Batch {batch_index + 1} failed after {max_attempts} attempts",
                    show_user=True
                )
                return (batch_index, "")


def process_ai_in_batches(
    words_list: List[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    card_format: Optional[CardFormat] = None
) -> Tuple[str, List[str]]:
    """Process words in batches using AI with progress reporting. Batches run concurrently.

    Returns:
        (content, failed_words): concatenated AI text and list of words whose batch failed.
    """
    if not words_list:
        return "", []

    client = get_openai_client()
    if not client:
        return "", []

    model_name = get_config()["openai_model"]
    total_words = len(words_list)
    batch_size = constants.AI_BATCH_SIZE
    concurrency = constants.AI_CONCURRENCY

    batches: List[Tuple[int, List[str]]] = []
    for i in range(0, total_words, batch_size):
        batch = words_list[i:i + batch_size]
        batches.append((len(batches), batch))

    results: List[Tuple[int, str]] = []
    failed_words: List[str] = []
    progress_lock = threading.Lock()
    completed_words = [0]  # mutable so inner fn can update

    def _on_batch_done(idx: int, content: str) -> None:
        results.append((idx, content))
        batch_word_count = len(batches[idx][1])
        with progress_lock:
            completed_words[0] += batch_word_count
            if not content:
                failed_words.extend(batches[idx][1])
        if progress_callback:
            progress_callback(min(completed_words[0], total_words), total_words)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _process_one_batch,
                idx, batch, client, model_name, card_format
            ): idx
            for idx, batch in batches
        }
        for future in as_completed(futures):
            try:
                idx, content = future.result()
                _on_batch_done(idx, content)
            except Exception as e:
                batch_idx = futures[future] + 1
                logger.error("Batch %s unexpected error: %s", batch_idx, e)
                ErrorHandler.handle(e, f"Batch {batch_idx} unexpected error", show_user=True)
                _on_batch_done(futures[future], "")

    results.sort(key=lambda x: x[0])
    return "\n".join(content for _, content in results if content), failed_words

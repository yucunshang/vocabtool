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
    """Build the built-in AI card generation prompt from card format settings."""
    if fmt is None:
        fmt = DEFAULT_CARD_FORMAT

    front_type = fmt.get("front", "word")
    def_lang = fmt.get("definition", "both")
    num_examples = fmt.get("examples", 3)
    include_ety = fmt.get("etymology", True)
    examples_with_cn = fmt.get("examples_with_cn", True)
    examples_colloquial = fmt.get("examples_colloquial", False)

    # Field 1: Word or Phrase
    if front_type == "phrase":
        f1_name = "Natural Phrase/Collocation"
        f1_constraint = "Output a high-frequency collocation or short phrase containing the target word (e.g. rain → heavy rain)."
        f1_example_altruism, f1_example_hectic = "motivated by altruism", "a hectic schedule"
    else:
        f1_name = "Target Word"
        f1_constraint = "Output the target word itself (lowercase). Do NOT add extra words."
        f1_example_altruism, f1_example_hectic = "altruism", "hectic"

    # Field 2: Definition
    if def_lang == "cn":
        f2_name = "Chinese Definition"
        f2_constraint = "Simplified Chinese only. Prefer 2–4 characters (e.g. 利他、忙乱)."
        f2_example_altruism, f2_example_hectic = "利他", "忙乱"
    elif def_lang == "en":
        f2_name = "English Definition"
        f2_constraint = "Concise English definition (B2-C1 level, under 15 words)."
        f2_example_altruism = "acting out of selfless concern for the well-being of others"
        f2_example_hectic = "full of frantic activity; very busy"
    elif def_lang == "en_native":
        f2_name = "English Definition (Native-Speaker Dictionary Style)"
        f2_constraint = "Use native-speaker dictionary style (e.g. Merriam-Webster, Oxford). May use advanced vocabulary."
        f2_example_altruism = "unselfish concern for the welfare of others"
        f2_example_hectic = "characterized by intense activity, confusion, or haste"
    else:  # both
        f2_name = "Chinese + English Definition"
        f2_constraint = "Format: `中文释义 / English definition`. Chinese: 2–8 chars. English: concise B2–C1."
        f2_example_altruism = "利他主义 / selfless concern for others"
        f2_example_hectic = "忙乱的 / full of frantic activity"

    # Field 3: Examples
    colloquial_note = " Use natural spoken/colloquial English." if examples_colloquial else ""
    ex_label = f"{num_examples} Example Sentence{'s' if num_examples > 1 else ''}"
    if num_examples == 1:
        f3_fmt = "ONE sentence" + (" with (中文翻译)" if examples_with_cn else "")
        f3_example_altruism = "His donation was motivated by altruism, not a desire for fame. (他的捐赠出于利他之心，而非求名。)" if examples_with_cn else "His donation was motivated by altruism, not a desire for fame."
        f3_example_hectic = "She has a hectic schedule with meetings all day. (她今天日程排满，会议不断。)" if examples_with_cn else "She has a hectic schedule with meetings all day."
    elif num_examples == 2:
        f3_fmt = "TWO sentences separated by ` // `" + ("; each with (中文翻译)" if examples_with_cn else "")
        f3_example_altruism = "His donation was motivated by altruism. (他的捐赠出于利他之心。) // True altruism expects nothing in return. (真正的利他主义不求回报。)" if examples_with_cn else "His donation was motivated by altruism. // True altruism expects nothing in return."
        f3_example_hectic = "She has a hectic schedule today. (她今天日程很满。) // The hectic pace can be exhausting. (快节奏令人疲惫。)" if examples_with_cn else "She has a hectic schedule today. // The hectic pace can be exhausting."
    else:
        f3_fmt = "THREE sentences separated by ` // `" + ("; each with (中文翻译)" if examples_with_cn else "")
        f3_example_altruism = "His donation was motivated by altruism. (他的捐赠出于利他之心。) // True altruism expects nothing in return. (真正的利他主义不求回报。) // Altruism is a core value. (利他主义是核心价值。)" if examples_with_cn else "His donation was motivated by altruism. // True altruism expects nothing in return. // Altruism is a core value."
        f3_example_hectic = "She has a hectic schedule today. (她今天日程很满。) // The hectic pace can be exhausting. (快节奏令人疲惫。) // After a hectic week, he relaxed. (忙了一周后他放松了。)" if examples_with_cn else "She has a hectic schedule today. // The hectic pace can be exhausting. // After a hectic week, he relaxed."
    f3_constraint = f"Exactly {f3_fmt}.{colloquial_note}"

    # Field 4: Deep Etymology & Roots (optional)
    if include_ety:
        f4_constraint = """**Deep Etymology & Roots (STRICT CONSTRAINTS)**

1. **Deep Classical Roots:** You MUST trace words back to their classical roots (Latin, Greek, Old English, etc.). DO NOT settle for shallow, modern morphological splits.
   - *Bad:* protectionist → protect (保护) + -ion (行为) + -ist (主义者)
   - *Good:* protectionist → pro- (向前/在前) + teg-/tect- (覆盖/掩蔽) + -ion (名词后缀) + -ist (主义者)

2. **Format:** Output strictly as: `prefix- (Chinese meaning) + root (Chinese meaning) + -suffix (Chinese meaning)`. Use Simplified Chinese for all meanings.

3. **Ban on Lazy Fallbacks:** DO NOT abuse "词源不可考" (Etymology unverified) as a shortcut. Words like synergy, prudence, latency have clear classical origins and MUST be broken down.

4. **Modern/Cultural Words:** For modern coinages, portmanteaus, eponyms (e.g., vegan, stoic, boycott), DO NOT label as unverified. Provide a concise 1–2 sentence origin story in Simplified Chinese.
   - *Example (vegan):* 现代造词：1944年由 veg(etari)an 的首尾字母拼接而成。

5. **Zero Hallucination:** Only if a word is genuinely of unknown origin, true slang, or purely onomatopoeic (e.g., clink, hiccup), output exactly: `词源不可考`. Never invent or fabricate roots."""
        f4_structure = " ||| `Etymology (Simplified Chinese)`"
        f4_example_altruism = " ||| alter- (其他) + -ism (主义/行为)"
        f4_example_hectic = " ||| hect- (持续的) + -ic (形容词后缀)"
    else:
        f4_constraint = ""
        f4_structure = ""
        f4_example_altruism = ""
        f4_example_hectic = ""

    field_constraints = "\n\n".join(filter(None, [
        f"1. **Field 1: {f1_name}**\n   - {f1_constraint}",
        f"2. **Field 2: {f2_name}**\n   - {f2_constraint}",
        f"3. **Field 3: Examples**\n   - {f3_constraint}",
        f"4. **Field 4: Roots/Etymology**\n   - {f4_constraint}" if include_ety else None,
    ]))

    if include_ety:
        mandatory_note = f"""
- Every line MUST have exactly 4 parts separated by `|||`: (1) word/phrase, (2) definition, (3) examples, (4) etymology.
- Field 3: exactly {num_examples} example sentence{'s' if num_examples > 1 else ''}. Separated by ` // `.
- Field 4: required. Deep classical roots; use `词源不可考` only when genuinely unknown/slang/onomatopoeic.
"""
    else:
        mandatory_note = f"""
- Every line MUST have exactly 3 parts separated by `|||`: (1) word/phrase, (2) definition, (3) examples. Do NOT add Field 4.
- Field 3: exactly {num_examples} example sentence{'s' if num_examples > 1 else ''}. Separated by ` // `.
"""

    return CARD_GEN_USER_TEMPLATE.format(
        mandatory_note=mandatory_note,
        words_str=words_str,
        f1_name=f1_name,
        f2_name=f2_name,
        ex_label=ex_label,
        f4_structure=f4_structure,
        field_constraints=field_constraints,
        f1_example_altruism=f1_example_altruism,
        f2_example_altruism=f2_example_altruism,
        f3_example_altruism=f3_example_altruism,
        f4_example_altruism=f4_example_altruism,
        f1_example_hectic=f1_example_hectic,
        f2_example_hectic=f2_example_hectic,
        f3_example_hectic=f3_example_hectic,
        f4_example_hectic=f4_example_hectic,
    )


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
    for attempt in range(constants.MAX_RETRIES):
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
            if attempt < constants.MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            else:
                ErrorHandler.handle(
                    e,
                    f"Batch {batch_index + 1} failed after {constants.MAX_RETRIES} attempts",
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
        if content:
            if progress_callback:
                with progress_lock:
                    completed_words[0] += len(batches[idx][1])
                    progress_callback(min(completed_words[0], total_words), total_words)
        else:
            with progress_lock:
                failed_words.extend(batches[idx][1])

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

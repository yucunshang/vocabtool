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
    """Build a dynamic AI prompt for Anki card generation based on card format."""
    if fmt is None:
        fmt = DEFAULT_CARD_FORMAT

    front_type = fmt.get("front", "phrase")
    def_lang = fmt.get("definition", "en")
    num_examples = fmt.get("examples", 1)
    include_ety = fmt.get("etymology", True)
    examples_with_cn = fmt.get("examples_with_cn", True)
    examples_colloquial = fmt.get("examples_colloquial", False)

    # ---- Field 1 description ----
    if front_type == "phrase":
        f1_name = "Natural Phrase/Collocation"
        f1_constraint = """1. **Field 1: Phrase (CRITICAL)**
   - DO NOT output the single target word alone.
   - You MUST generate a high-frequency **collocation** or **short phrase** containing the target word.
   - Example: If input is "rain", output "heavy rain" or "torrential rain"."""
        f1_example_altruism = "motivated by altruism"
        f1_example_hectic = "a hectic schedule"
    else:
        f1_name = "Target Word"
        f1_constraint = """1. **Field 1: Word**
   - Output the target word itself (lowercase).
   - Do NOT add extra words or phrases."""
        f1_example_altruism = "altruism"
        f1_example_hectic = "hectic"

    # ---- Field 2 description ----
    if def_lang == "cn":
        f2_name = "Chinese Definition"
        f2_constraint = """2. **Field 2: Definition (中文)**
   - Simplified Chinese only. Prefer a SINGLE word or 2-4 characters (e.g. 利他、忙乱). Avoid long phrases."""
        f2_example_altruism = "利他"
        f2_example_hectic = "忙乱"
    elif def_lang == "en":
        f2_name = "English Definition"
        f2_constraint = """2. **Field 2: Definition (English)**
   - Concise English definition (B2-C1 level, under 15 words)."""
        f2_example_altruism = "acting out of selfless concern for the well-being of others"
        f2_example_hectic = "full of frantic activity; very busy"
    elif def_lang == "en_native":
        f2_name = "English Definition (Native-Speaker Dictionary Style)"
        f2_constraint = """2. **Field 2: Definition (Native-Speaker English)**
   - Use native-speaker dictionary style (e.g. Merriam-Webster, Oxford English Dictionary).
   - May use advanced vocabulary for precision. Concise but NOT restricted to learner/defining vocabulary."""
        f2_example_altruism = "unselfish concern for the welfare of others"
        f2_example_hectic = "characterized by intense activity, confusion, or haste"
    else:
        f2_name = "Chinese + English Definition"
        f2_constraint = """2. **Field 2: Definition (中英双语)**
   - Format: `中文释义 / English definition`
   - Chinese part: 2-8 characters. English part: concise B2-C1 level."""
        f2_example_altruism = "利他主义 / selfless concern for others"
        f2_example_hectic = "忙乱的 / full of frantic activity"

    # ---- Field 3 description (examples) ----
    colloquial_note = "\n   - Use natural **spoken/colloquial** English (口语化); like daily conversation. Avoid formal or written style." if examples_colloquial else ""
    ex_label = f"{num_examples} Example Sentence{'s' if num_examples > 1 else ''}"
    if num_examples == 1:
        if examples_with_cn:
            f3_constraint = """3. **Field 3: Example**
   - ONE short, authentic English sentence, then ` (中文翻译)` on the same line.""" + colloquial_note
            f3_example_altruism = "His donation was motivated by altruism, not a desire for fame. (他的捐赠出于利他之心，而非求名。)"
            f3_example_hectic = "She has a hectic schedule with meetings all day. (她今天日程排满，会议不断。)"
        else:
            f3_constraint = """3. **Field 3: Example**
   - ONE short, authentic English sentence containing the word/phrase. No Chinese translation.""" + colloquial_note
            f3_example_altruism = "His donation was motivated by altruism, not a desire for fame."
            f3_example_hectic = "She has a hectic schedule with meetings all day."
    elif num_examples == 2:
        if examples_with_cn:
            f3_constraint = """3. **Field 3: Examples (2 sentences, each with Chinese translation)**
   - TWO short, authentic English sentences separated by ` // `.
   - Each segment MUST be: `English sentence (中文翻译)` — same line, parentheses with Chinese.
   - Each English sentence must contain the target word/phrase.""" + colloquial_note
            f3_example_altruism = "His donation was motivated by altruism, not a desire for fame. (他的捐赠出于利他之心，而非求名。) // True altruism expects nothing in return. (真正的利他主义不求回报。)"
            f3_example_hectic = "She has a hectic schedule with meetings all day. (她今天日程排满，会议不断。) // The hectic pace of city life can be exhausting. (城市生活的快节奏令人疲惫。)"
        else:
            f3_constraint = """3. **Field 3: Examples (2 sentences)**
   - TWO short, authentic English sentences separated by ` // `. No Chinese translation.""" + colloquial_note
            f3_example_altruism = "His donation was motivated by altruism, not a desire for fame. // True altruism expects nothing in return."
            f3_example_hectic = "She has a hectic schedule with meetings all day. // The hectic pace of city life can be exhausting."
    else:
        if examples_with_cn:
            f3_constraint = """3. **Field 3: Examples (3 sentences, each with Chinese translation)**
   - THREE short, authentic English sentences separated by ` // `. Each: `English (中文)`.""" + colloquial_note
            f3_example_altruism = "His donation was motivated by altruism. (他的捐赠出于利他之心。) // True altruism expects nothing in return. (真正的利他主义不求回报。) // Altruism is a core value. (利他主义是核心价值。)"
            f3_example_hectic = "She has a hectic schedule today. (她今天日程很满。) // The hectic pace can be exhausting. (快节奏令人疲惫。) // After a hectic week, he relaxed. (忙了一周后他放松了。)"
        else:
            f3_constraint = """3. **Field 3: Examples (3 sentences)**
   - THREE short, authentic English sentences separated by ` // `. No Chinese.""" + colloquial_note
            f3_example_altruism = "His donation was motivated by altruism. // True altruism expects nothing in return. // Altruism is a core value in many cultures."
            f3_example_hectic = "She has a hectic schedule today. // The hectic pace of city life can be exhausting. // After a hectic week, he finally relaxed."

    # ---- Field 4 description (etymology, optional) ----
    if include_ety:
        f4_constraint = """4. **Field 4: Roots/Etymology (Simplified Chinese)**
   - Format: `prefix- (meaning) + root (meaning) + -suffix (meaning)`.
   - If no classical roots exist, explain the origin briefly in Chinese.
   - Use Simplified Chinese for meanings."""
        f4_example_altruism = " ||| alter (其他) + -ism (主义/行为)"
        f4_example_hectic = " ||| hect- (持续的/习惯性的 - 来自希腊语hektikos) + -ic (形容词后缀)"
        f4_structure = " ||| `Etymology breakdown (Simplified Chinese)`"
        f4_label = ", Etymology"
    else:
        f4_constraint = ""
        f4_example_altruism = ""
        f4_example_hectic = ""
        f4_structure = ""
        f4_label = ""

    # ---- Assemble ----
    field_constraints = "\n\n".join(filter(None, [f1_constraint, f2_constraint, f3_constraint, f4_constraint]))

    mandatory_note = ""
    if num_examples >= 2:
        if include_ety:
            mandatory_note = """
# CRITICAL (Do not skip)
- Every line MUST have exactly 4 parts separated by `|||`: (1) word/phrase, (2) definition, (3) examples, (4) etymology.
- Field 3: exactly 2 example sentences (with or without Chinese per format). Separated by ` // `.
- Field 4: etymology/roots in Chinese is REQUIRED for every word. If uncertain, give a brief origin note in Chinese.
"""
        else:
            mandatory_note = """
# CRITICAL (Do not skip)
- Every line MUST have exactly 3 parts separated by `|||`: (1) word/phrase, (2) definition, (3) examples. Do NOT add Field 4.
- Field 3: exactly 2 example sentences. Separated by ` // `.
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
                temperature=0.7
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
) -> str:
    """Process words in batches using AI with progress reporting. Batches run concurrently."""
    if not words_list:
        return ""

    client = get_openai_client()
    if not client:
        return ""

    model_name = get_config()["openai_model"]
    total_words = len(words_list)
    batch_size = constants.AI_BATCH_SIZE
    concurrency = constants.AI_CONCURRENCY

    batches: List[Tuple[int, List[str]]] = []
    for i in range(0, total_words, batch_size):
        batch = words_list[i:i + batch_size]
        batches.append((len(batches), batch))

    results: List[Tuple[int, str]] = []
    progress_lock = threading.Lock()
    completed_words = [0]  # mutable so inner fn can update

    def _on_batch_done(idx: int, content: str) -> None:
        results.append((idx, content))
        if progress_callback and content:
            with progress_lock:
                completed_words[0] += len(batches[idx][1])
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
    return "\n".join(content for _, content in results if content)

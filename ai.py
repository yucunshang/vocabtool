# AI-backed word definitions and batch card generation.
# 提示词模板见 prompts.py，可直接修改，不影响制卡 / apkg / 语音 / 卡片格式。

import logging
import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import constants
from config import get_config
from errors import ErrorHandler
from persistent_cache import cache_get as persistent_cache_get, cache_set as persistent_cache_set
from prompts import (
    CARD_GEN_CLOZE_TEMPLATE,
    CARD_GEN_PRODUCTION_TEMPLATE,
    CARD_GEN_SYSTEM_PROMPT,
    CARD_GEN_TRANSLATION_TEMPLATE,
    CARD_GEN_USER_TEMPLATE,
    LOOKUP_SYSTEM_PROMPT,
    THIRD_PARTY_CARD_TEMPLATE,
    THIRD_PARTY_CLOZE_TEMPLATE,
    THIRD_PARTY_TRANSLATION_TEMPLATE,
    THIRD_PARTY_PRODUCTION_TEMPLATE,
    THIRD_PARTY_AUDIO_TEMPLATE,
)
from resources import get_rank_for_word

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("OpenAI library not available")


class CardFormat(TypedDict, total=False):
    card_type: str    # "standard" | "cloze" | "production" | "translation"
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
_CARD_BATCH_CACHE: OrderedDict[str, str] = OrderedDict()
_CARD_BATCH_CACHE_MAX = 1000
_CARD_BATCH_CACHE_LOCK = threading.Lock()
_OPENAI_CLIENT: Optional[Any] = None


def _stable_cache_key(kind: str, model_name: str, prompt_text: str, payload: str) -> str:
    """Build a stable hash key for persistent cache entries."""
    raw = "\n".join([
        constants.APP_RELEASE_CHANNEL,
        kind,
        model_name,
        prompt_text,
        payload,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _set_lookup_cache(cache_key: str, content: str) -> None:
    _QUERY_CACHE[cache_key] = content
    _QUERY_CACHE.move_to_end(cache_key)
    if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
        _QUERY_CACHE.popitem(last=False)


def _get_card_batch_cache(cache_key: str) -> Optional[str]:
    with _CARD_BATCH_CACHE_LOCK:
        if cache_key in _CARD_BATCH_CACHE:
            _CARD_BATCH_CACHE.move_to_end(cache_key)
            return _CARD_BATCH_CACHE[cache_key]
    return None


def _set_card_batch_cache(cache_key: str, content: str) -> None:
    with _CARD_BATCH_CACHE_LOCK:
        _CARD_BATCH_CACHE[cache_key] = content
        _CARD_BATCH_CACHE.move_to_end(cache_key)
        if len(_CARD_BATCH_CACHE) > _CARD_BATCH_CACHE_MAX:
            _CARD_BATCH_CACHE.popitem(last=False)


def build_card_prompt(
    words_str: str,
    fmt: Optional[CardFormat] = None,
    card_type: Optional[str] = None,
) -> str:
    """Build the built-in AI card generation prompt by card type."""
    ct = card_type if card_type is not None else (fmt.get("card_type", "standard") if fmt else "standard")
    templates = {
        "standard":    CARD_GEN_USER_TEMPLATE,
        "cloze":       CARD_GEN_CLOZE_TEMPLATE,
        "production":  CARD_GEN_PRODUCTION_TEMPLATE,
        "translation": CARD_GEN_TRANSLATION_TEMPLATE,
        "audio":       CARD_GEN_USER_TEMPLATE,
    }
    tpl = templates.get(ct, CARD_GEN_USER_TEMPLATE)
    voice = (fmt or {}).get("voice_code", "")
    ipa_style = "British IPA" if voice.startswith("en-GB") else "American IPA"
    return tpl.format(words_str=words_str, ipa_style=ipa_style)


def build_thirdparty_prompt(words_str: str, fmt: Optional[CardFormat] = None) -> str:
    """Build third-party AI prompt from card format (up to 200 words per batch)."""
    if fmt is None:
        fmt = {
            "card_type": "standard",
            "front": "phrase",
            "definition": "en_native",
            "examples": 2,
            "examples_with_cn": False,
            "etymology": True,
        }

    card_type = fmt.get("card_type", "standard")
    if card_type == "cloze":
        voice = fmt.get("voice_code", "")
        ipa_style = "British IPA" if voice.startswith("en-GB") else "American IPA"
        return THIRD_PARTY_CLOZE_TEMPLATE.format(words_str=words_str, ipa_style=ipa_style)

    if card_type == "translation":
        return THIRD_PARTY_TRANSLATION_TEMPLATE.format(words_str=words_str)

    if card_type == "production":
        return THIRD_PARTY_PRODUCTION_TEMPLATE.format(words_str=words_str)

    if card_type == "audio":
        return THIRD_PARTY_AUDIO_TEMPLATE.format(words_str=words_str)

    front = fmt.get("front", "phrase")
    def_lang = fmt.get("definition", "en_native")
    num_ex = fmt.get("examples", 2)
    with_cn = fmt.get("examples_with_cn", False)
    include_ety = fmt.get("etymology", True)

    # Field 1
    if front == "phrase":
        field1_instruction = """1. **Field 1: Phrase (CRITICAL)**
   - DO NOT output the single target word.
   - You MUST generate a high-frequency **collocation** or **short phrase** containing the target word.
   - Example: If input is "rain", output "heavy rain" or "torrential rain"."""
    else:
        field1_instruction = """1. **Field 1: Word**
   - Output the target word in lowercase. No extra text."""

    # Field 2
    if def_lang == "cn":
        field2_instruction = """2. **Field 2: Definition (Chinese)**
   - ONE concise Chinese definition only. No slashes, no English."""
    elif def_lang == "en":
        field2_instruction = """2. **Field 2: Definition (English)**
   - Define the phrase/word. Keep it concise (B2-C1 learner dictionary style)."""
    elif def_lang == "en_native":
        field2_instruction = """2. **Field 2: Definition (English)**
   - Define the *phrase* or word in native-speaker dictionary style (e.g. Merriam-Webster, Oxford). Keep it concise."""
    else:
        field2_instruction = """2. **Field 2: Definition (Bilingual)**
   - Format: `中文释义 / English definition`. Both required."""

    # Field 3
    ex_label = f"{num_ex} example sentence{'s' if num_ex > 1 else ''}"
    if with_cn:
        field3_instruction = f"""3. **Field 3: Example{'s' if num_ex > 1 else ''}**
   - {ex_label.capitalize()}. Each MUST include a Chinese translation: `English sentence. (中文翻译。)` Separate with ` // `."""
    else:
        field3_instruction = f"""3. **Field 3: Example{'s' if num_ex > 1 else ''}**
   - {ex_label.capitalize()}. English only, NO Chinese translation. Separate with ` // `."""

    # Field 4
    if include_ety:
        field4_instruction = """4. **Field 4: Roots/Etymology (Simplified Chinese)**
   - Format: `prefix- (meaning) + root (meaning) + -suffix (meaning)`.
   - If no classical roots exist, explain the origin briefly in Chinese. Use "词源不可考" only when genuinely unknown."""
        structure_line = "`Field1` ||| `Field2` ||| `Field3` ||| `Field4 (Etymology)`"
    else:
        field4_instruction = ""
        structure_line = "`Field1` ||| `Field2` ||| `Field3`"

    # Example line
    if front == "phrase" and def_lang == "en_native" and num_ex >= 1 and not with_cn and include_ety:
        example_line = """Input: altruism
Output:
motivated by altruism ||| acting out of selfless concern for the well-being of others ||| His donation was motivated by altruism, not a desire for fame. ||| alter (其他) + -ism (主义/行为)

Input: hectic
Output:
a hectic schedule ||| a timeline full of frantic activity and very busy ||| She has a hectic schedule with meetings all day. ||| hect- (持续的 - 来自希腊语hektikos) + -ic (形容词后缀)"""
    else:
        example_line = "Input: word\nOutput:\nphrase_or_word ||| definition ||| Example sentence." + (" ||| 词源" if include_ety else "")

    return THIRD_PARTY_CARD_TEMPLATE.format(
        field1_instruction=field1_instruction,
        field2_instruction=field2_instruction,
        field3_instruction=field3_instruction,
        field4_instruction=field4_instruction,
        structure_line=structure_line,
        example_line=example_line,
        words_str=words_str,
    )


def build_thirdparty_format_definition(fmt: Optional[CardFormat] = None) -> str:
    """Build a concise format-definition spec for third-party AI (all card types)."""
    if fmt is None:
        fmt = {
            "card_type": "standard",
            "front": "phrase",
            "definition": "en_native",
            "examples": 2,
            "examples_with_cn": False,
            "etymology": True,
        }

    card_type = fmt.get("card_type", "standard")
    lines: List[str] = [
        "【第三方 AI 卡片格式定义】",
        "总量不限；系统会自动按每批最多 200 词分组。",
        "每行一张卡，字段分隔符固定为：|||",
        "字段内禁止再次出现分隔符 |||",
    ]

    if card_type == "cloze":
        lines.extend([
            "卡片类型：阅读卡（cloze）",
            "字段格式：挖空句 ||| Target word /IPA/ pos. 中文释义 ||| 完整句(含中文翻译)",
            "约束：挖空位固定为 ________（8个下划线）；3字段固定。",
            "示例：The doorknob was made of polished ________ and reflected the hallway light. ||| brass /bræs/ n. 黄铜 ||| The doorknob was made of polished brass and reflected the hallway light. (门把手由抛光黄铜制成，反射着走廊里的灯光。)",
        ])
        return "\n".join(lines)

    if card_type == "translation":
        lines.extend([
            "卡片类型：互译卡（translation）",
            "字段格式：中文释义 ||| English word / IPA ||| 例句(含中文翻译)",
            "约束：3字段固定；第2字段必须包含英文词头与音标。",
            "示例：模糊的，含混不清的 ||| ambiguous / æmˈbɪɡjuəs ||| The instructions were ambiguous. (说明含糊不清。)",
        ])
        return "\n".join(lines)

    if card_type == "production":
        lines.extend([
            "卡片类型：表达卡（production）",
            "字段格式：中文场景描述 ||| English chunk / collocation ||| 例句(含中文翻译)",
            "约束：3字段固定；第2字段建议为高可复用词块/搭配。",
            "示例：你想说：这份声明措辞模糊。 ||| ambiguous statement ||| The government's ambiguous statement caused confusion. (政府含糊其辞的声明引发困惑。)",
        ])
        return "\n".join(lines)

    if card_type == "audio":
        lines.extend([
            "卡片类型：听音卡（audio）",
            "字段格式：English word/phrase ||| 中英释义 ||| 例句(含中文翻译)",
            "约束：3字段固定；第1字段仅保留词/短语本体。",
            "示例：heavy rain ||| 大雨 | intense rainfall ||| We got caught in heavy rain. (我们遇上了大雨。)",
        ])
        return "\n".join(lines)

    front = fmt.get("front", "phrase")
    def_lang = fmt.get("definition", "en_native")
    num_ex = int(fmt.get("examples", 2))
    with_cn = bool(fmt.get("examples_with_cn", False))
    include_ety = bool(fmt.get("etymology", True))

    if front == "phrase":
        f1 = "Field1=短语/搭配（必须包含目标词，不可只输出单词）"
    else:
        f1 = "Field1=目标单词（小写，纯词头）"

    if def_lang == "cn":
        f2 = "Field2=中文释义（仅中文）"
    elif def_lang == "en":
        f2 = "Field2=英文释义（学习型）"
    elif def_lang == "en_native":
        f2 = "Field2=英文释义（母语词典风格）"
    else:
        f2 = "Field2=中英双语释义（中文 / English）"

    ex_label = f"{num_ex} 条例句"
    if with_cn:
        f3 = f"Field3={ex_label}（每条需中文翻译，用 // 分隔）"
    else:
        f3 = f"Field3={ex_label}（英文，仅 // 分隔）"

    if include_ety:
        structure = "Field1 ||| Field2 ||| Field3 ||| Field4"
        f4 = "Field4=词根词源/构词说明（中文）"
    else:
        structure = "Field1 ||| Field2 ||| Field3"
        f4 = "Field4=无"

    lines.extend([
        "卡片类型：标准卡（standard，可自定义）",
        f"目标结构：{structure}",
        f1,
        f2,
        f3,
        f4,
        "示例：a hectic schedule ||| very busy and full of activity ||| She has a hectic schedule with meetings all day. ||| hect- + -ic",
    ])
    return "\n".join(lines)


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
        if not line or line.startswith(('✏️ 拼写纠正:', '✔️ 拼写纠正:', '拼写纠正:')):
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

    persistent_key = _stable_cache_key(
        kind="lookup",
        model_name=model_name,
        prompt_text=LOOKUP_SYSTEM_PROMPT,
        payload=word_clean,
    )
    persistent_cached = persistent_cache_get(persistent_key)
    if persistent_cached:
        _set_lookup_cache(cache_key, persistent_cached)
        return {"result": persistent_cached, "rank": _rank_from_ai_content(persistent_cached, rank), "cached": True}

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
                    delta_obj = event.choices[0].delta
                    delta = (delta_obj.content or "") if delta_obj is not None else ""
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

        _set_lookup_cache(cache_key, content)
        persistent_cache_set(persistent_key, content)

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
    cache_key = _stable_cache_key(
        kind="card_batch",
        model_name=model_name,
        prompt_text=CARD_GEN_SYSTEM_PROMPT,
        payload=user_prompt,
    )

    cached_content = _get_card_batch_cache(cache_key)
    if cached_content:
        return (batch_index, cached_content)

    persistent_cached = persistent_cache_get(cache_key)
    if persistent_cached:
        _set_card_batch_cache(cache_key, persistent_cached)
        return (batch_index, persistent_cached)

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
            _set_card_batch_cache(cache_key, content)
            persistent_cache_set(cache_key, content)
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

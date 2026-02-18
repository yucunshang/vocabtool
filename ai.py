# AI-backed word definitions and batch card generation.

import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, TypedDict

import streamlit as st

import constants
from config import get_config
from errors import ErrorHandler
from resources import get_rank_for_word

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("OpenAI library not available")


class CardFormat(TypedDict, total=False):
    front: str        # "word" | "phrase"
    definition: str   # "cn" | "en" | "both"
    examples: int     # 1 | 2 | 3
    etymology: bool   # True | False


# Built-in card format: front=word, back=中文释义 + 2例句(含中文翻译) + 词根词源，每例句单独音频
DEFAULT_CARD_FORMAT: CardFormat = {
    "front": "word",
    "definition": "cn",
    "examples": 2,
    "etymology": True,
}

# Fast in-memory cache for quick lookup to match vocabtool behavior.
_QUERY_CACHE: OrderedDict[str, str] = OrderedDict()
_QUERY_CACHE_MAX = 500
_OPENAI_CLIENT: Optional[Any] = None

LOOKUP_SYSTEM_PROMPT = """# Role
Atomic Flash Dictionary.

# Goal
Provide the SINGLE most common meaning of a word in a strict 5-line format with clean POS tags.

# Critical Constraints
1.  **Force Single Sense**: Pick ONLY the #1 most common meaning/POS combination.
2.  **Strict Alignment**: The Definition, Etymology, and BOTH Examples must strictly refer to this ONE meaning.
3.  **Formatting**:
    - **Line 1**: `[word] ([pos] [CN pos])` (No dots, no commas).
    - **No Markdown**: Pure text only.
    - **Compactness**: Example and Translation must be on the SAME line.

# Output Format
[word] ([pos] [CN pos])
[CN Meaning] | [Short EN Definition (<8 words)]
🌱 词源: [root (CN) + affix (CN)] (Or brief origin)
• [English Example 1] ([CN Trans])
• [English Example 2] ([CN Trans])

# Few-Shot Examples (Visual Style: Clean)
**User Input:**
spring

**Model Output:**
spring (n 名词)
春天 | The season after winter
🌱 词源: spring- (涌出/生长) → 万物复苏的季节
• Flowers bloom in spring. (花朵在春天绽放。)
• I love the fresh air of spring. (我喜欢春天清新的空气。)

**User Input:**
date

**Model Output:**
date (n 名词)
日期 | Specific day of the month
🌱 词源: dat- (给予/指定) + -e (名词后缀)
• What is today's date? (今天是几号？)
• Please sign and date the form. (请在表格上签名并注明日期。)

**User Input:**
express

**Model Output:**
express (v 动词)
表达；表示 | Convey a thought or feeling
🌱 词源: ex- (向外) + press (压/挤)
• She expressed her thanks to us. (她向我们表达了谢意。)
• Words cannot express my feelings. (言语无法表达我的感受。)"""


def build_card_prompt(words_str: str, fmt: Optional[CardFormat] = None) -> str:
    """Build a dynamic AI prompt for Anki card generation based on card format."""
    if fmt is None:
        fmt = DEFAULT_CARD_FORMAT

    front_type = fmt.get("front", "phrase")
    def_lang = fmt.get("definition", "en")
    num_examples = fmt.get("examples", 1)
    include_ety = fmt.get("etymology", True)

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
   - Simplified Chinese only. Concise meaning (2-8 characters preferred)."""
        f2_example_altruism = "利他主义/无私"
        f2_example_hectic = "忙乱的/繁忙的"
    elif def_lang == "en":
        f2_name = "English Definition"
        f2_constraint = """2. **Field 2: Definition (English)**
   - Concise English definition (B2-C1 level, under 15 words)."""
        f2_example_altruism = "acting out of selfless concern for the well-being of others"
        f2_example_hectic = "full of frantic activity; very busy"
    else:
        f2_name = "Chinese + English Definition"
        f2_constraint = """2. **Field 2: Definition (中英双语)**
   - Format: `中文释义 / English definition`
   - Chinese part: 2-8 characters. English part: concise B2-C1 level."""
        f2_example_altruism = "利他主义 / selfless concern for others"
        f2_example_hectic = "忙乱的 / full of frantic activity"

    # ---- Field 3 description (examples) ----
    ex_label = f"{num_examples} Example Sentence{'s' if num_examples > 1 else ''}"
    if num_examples == 1:
        f3_constraint = """3. **Field 3: Example**
   - ONE short, authentic English sentence containing the word/phrase."""
        f3_example_altruism = "His donation was motivated by altruism, not a desire for fame."
        f3_example_hectic = "She has a hectic schedule with meetings all day."
    elif num_examples == 2:
        f3_constraint = """3. **Field 3: Examples (2 sentences, each with Chinese translation)**
   - TWO short, authentic English sentences separated by ` // `.
   - Each segment MUST be: `English sentence (中文翻译)` — same line, parentheses with Chinese.
   - Each English sentence must contain the target word/phrase."""
        f3_example_altruism = "His donation was motivated by altruism, not a desire for fame. (他的捐赠出于利他之心，而非求名。) // True altruism expects nothing in return. (真正的利他主义不求回报。)"
        f3_example_hectic = "She has a hectic schedule with meetings all day. (她今天日程排满，会议不断。) // The hectic pace of city life can be exhausting. (城市生活的快节奏令人疲惫。)"
    else:
        f3_constraint = """3. **Field 3: Examples (3 sentences)**
   - THREE short, authentic English sentences separated by ` // `.
   - Each sentence must contain the target word/phrase."""
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
    if include_ety and num_examples >= 2:
        mandatory_note = """
# CRITICAL (Do not skip)
- Every line MUST have exactly 4 parts separated by `|||`: (1) word/phrase, (2) definition, (3) examples, (4) etymology.
- Field 3: exactly 2 example sentences, each with Chinese translation in parentheses, separated by ` // `.
- Field 4: etymology/roots in Chinese is REQUIRED for every word. If uncertain, give a brief origin note in Chinese.
"""

    prompt = f"""# Role
You are an expert English Lexicographer and Anki Card Designer. Your goal is to convert a list of target words into high-quality, import-ready Anki flashcards.
Make sure to process everything in one go, without missing anything.
{mandatory_note}
# Input Data
{words_str}

# Output Format Guidelines
1. **Output Container**: Strictly inside a single ```text code block.
2. **Layout**: One entry per line.
3. **Separator**: Use `|||` as the delimiter.
4. **Target Structure** (every line must have all parts):
   `{f1_name}` ||| `{f2_name}` ||| `{ex_label}`{f4_structure}

# Field Constraints (Strict)
{field_constraints}

# Valid Example (Follow this logic strictly)
Input: altruism
Output:
{f1_example_altruism} ||| {f2_example_altruism} ||| {f3_example_altruism}{f4_example_altruism}

Input: hectic
Output:
{f1_example_hectic} ||| {f2_example_hectic} ||| {f3_example_hectic}{f4_example_hectic}

# Task
Process the provided input list strictly adhering to the format above."""

    return prompt


def get_openai_client() -> Optional[Any]:
    """Get configured OpenAI client with proper error handling."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    if not OpenAI:
        st.error("❌ 未安装 OpenAI 库，无法使用内置 AI 功能。")
        return None

    cfg = get_config()
    api_key = cfg["openai_api_key"]
    if not api_key:
        st.error("❌ 未找到 OPENAI_API_KEY。请在 .streamlit/secrets.toml 中配置。")
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


def get_word_quick_definition(
    word: str,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Get ultra-concise word definition using AI, with rank info.

    If ``stream_callback`` is provided, tokens are streamed incrementally.
    """
    word_lower = word.lower().strip()
    rank = get_rank_for_word(word)

    client = get_openai_client()
    if not client:
        return {"error": "AI client not available"}

    model_name = get_config()["openai_model"]

    cache_key = word_lower
    if cache_key in _QUERY_CACHE:
        _QUERY_CACHE.move_to_end(cache_key)
        return {"result": _QUERY_CACHE[cache_key], "rank": rank, "cached": True}

    try:
        if stream_callback is not None:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": LOOKUP_SYSTEM_PROMPT},
                    {"role": "user", "content": word}
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
                    {"role": "user", "content": word}
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

        return {"result": content, "rank": rank}

    except Exception as e:
        logger.error("Error getting definition: %s", e)
        return {"error": str(e)}


def process_ai_in_batches(
    words_list: List[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    card_format: Optional[CardFormat] = None
) -> Optional[str]:
    """Process words in batches using AI with progress reporting."""
    if not words_list:
        return ""

    client = get_openai_client()
    if not client:
        return None

    model_name = get_config()["openai_model"]
    total_words = len(words_list)
    full_results: List[str] = []

    system_prompt = "You are a helpful assistant for vocabulary learning."

    for i in range(0, total_words, constants.AI_BATCH_SIZE):
        batch = words_list[i:i + constants.AI_BATCH_SIZE]
        current_batch_str = ", ".join(batch)

        user_prompt = build_card_prompt(current_batch_str, card_format)

        for attempt in range(constants.MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError("Empty AI batch response")
                full_results.append(content)

                if progress_callback:
                    processed_count = min(i + constants.AI_BATCH_SIZE, total_words)
                    progress_callback(processed_count, total_words)

                break

            except Exception as e:
                if attempt < constants.MAX_RETRIES - 1:
                    time.sleep(1 + attempt)
                    continue
                else:
                    ErrorHandler.handle(
                        e,
                        f"Batch {i//constants.AI_BATCH_SIZE + 1} failed after {constants.MAX_RETRIES} attempts",
                        show_user=True
                    )

    return "\n".join(full_results)

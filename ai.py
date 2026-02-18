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


DEFAULT_CARD_FORMAT: CardFormat = {
    "front": "word",
    "definition": "cn",
    "examples": 1,
    "etymology": False,
}

# Fast in-memory cache for quick lookup to match vocabtool behavior.
_QUERY_CACHE: OrderedDict[str, str] = OrderedDict()
_QUERY_CACHE_MAX = 500
_OPENAI_CLIENT: Optional[Any] = None

# -----------------------------------------------------------------------------
# Consistent Lookup Prompts
# -----------------------------------------------------------------------------

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
"""


def build_card_prompt(words_str: str, fmt: Optional[CardFormat] = None) -> str:
    """
    Build a dynamic AI prompt for Anki card generation.
    matches the visual style of the single-word lookup tool (Atomic Flash Dictionary).
    """
    if fmt is None:
        fmt = DEFAULT_CARD_FORMAT

    front_type = fmt.get("front", "phrase")
    def_lang = fmt.get("definition", "en")
    num_examples = fmt.get("examples", 1)
    include_ety = fmt.get("etymology", True)

    # 1. Front Field Logic
    if front_type == "phrase":
        front_instruction = "Output a high-frequency short phrase/collocation containing the word."
    else:
        front_instruction = "Output the target word itself (clean, lowercase)."

    # 2. Definition Logic
    if def_lang == "cn":
        def_instruction = "[CN Meaning]"
    elif def_lang == "en":
        def_instruction = "[Short EN Definition (<12 words)]"
    else:
        def_instruction = "[CN Meaning] | [Short EN Definition]"

    # 3. Etymology Logic
    if include_ety:
        ety_instruction = "🌱 词源: [root (CN) + affix (CN)] (Briefly explain origin in Simplified Chinese) <br>"
    else:
        ety_instruction = ""  # Empty string if disabled

    # 4. Examples Logic
    # We construct the example template based on count
    example_lines = []
    for i in range(num_examples):
        example_lines.append(f"• [Example {i+1}] ([CN Trans])")
    
    # Join examples with <br> for HTML rendering in Anki
    examples_instruction = " <br> ".join(example_lines)

    prompt = f"""# Role
You are an expert Anki Card Generator.
Your task is to convert a list of words into Anki cards that perfectly match the "Atomic Flash Dictionary" visual style.

# Input Data
{words_str}

# Output Format (Strict CSV-like)
- **One line per entry**.
- **Separator**: `|||`
- **Structure**: `Field1_Front ||| Field2_Back`
- **Important**: For line breaks INSIDE the card back, use the HTML tag `<br>`. Do NOT use actual newlines inside the content, or the import will fail.

# Field 1: Front
{front_instruction}

# Field 2: Back (Visual Content)
Construct the back content strictly using this layout (use `<br>` for new lines):
`([pos] [CN pos]) <br>`
`{def_instruction} <br>`
`{ety_instruction}`
`{examples_instruction}`

# Critical Constraints
1. **Consistency**: The definition, etymology, and examples must all align with the SINGLE most common meaning of the word.
2. **Conciseness**: Definitions should be short. Example translations must be on the same line as the English sentence.
3. **Language**: Use Simplified Chinese for translations/meanings unless specified otherwise.
4. **No Markdown**: Do not use bold/italic markdown (** or *) in the output, just plain text with `<br>` tags.

# Example Output (Mock)
altruism ||| (n 名词) <br> 利他主义 | selfless concern for others <br> 🌱 词源: alter (其他) + -ism (主义) <br> • His donation was motivated by altruism. (他的捐款是出于利他主义。)
hectic ||| (adj 形容词) <br> 忙乱的 | full of frantic activity <br> 🌱 词源: hect- (习惯性) + -ic (后缀) <br> • She has a hectic schedule. (她的日程安排非常繁忙。)

# Task
Process the input list now.
"""
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

    # System prompt is now handled per-batch via build_card_prompt,
    # but we set a generic role here for the message history context.
    system_prompt = "You are a precise Anki card generator."

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
                
                # Remove markdown code blocks if AI adds them (e.g. ```text ... ```)
                content = content.replace("```text", "").replace("```", "").strip()
                
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
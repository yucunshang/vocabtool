# AI-backed word definitions and batch card generation.

import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypedDict

import streamlit as st

import constants
from config import get_config
from errors import ErrorHandler
from resources import get_vocab_dict

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
    "front": "phrase",
    "definition": "en",
    "examples": 1,
    "etymology": True,
}


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
        f3_constraint = """3. **Field 3: Examples (2 sentences)**
   - TWO short, authentic English sentences separated by ` // `.
   - Each sentence must contain the target word/phrase."""
        f3_example_altruism = "His donation was motivated by altruism, not a desire for fame. // True altruism expects nothing in return."
        f3_example_hectic = "She has a hectic schedule with meetings all day. // The hectic pace of city life can be exhausting."
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

    prompt = f"""# Role
You are an expert English Lexicographer and Anki Card Designer. Your goal is to convert a list of target words into high-quality, import-ready Anki flashcards.
Make sure to process everything in one go, without missing anything.

# Input Data
{words_str}

# Output Format Guidelines
1. **Output Container**: Strictly inside a single ```text code block.
2. **Layout**: One entry per line.
3. **Separator**: Use `|||` as the delimiter.
4. **Target Structure**:
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
    if not OpenAI:
        st.error("❌ 未安装 OpenAI 库，无法使用内置 AI 功能。")
        return None

    cfg = get_config()
    api_key = cfg["openai_api_key"]
    if not api_key:
        st.error("❌ 未找到 OPENAI_API_KEY。请在 .streamlit/secrets.toml 中配置。")
        return None

    try:
        return OpenAI(api_key=api_key, base_url=cfg["openai_base_url"])
    except Exception as e:
        ErrorHandler.handle(e, "Failed to initialize OpenAI client")
        return None


def get_word_quick_definition(word: str) -> Dict[str, Any]:
    """Get ultra-concise word definition using AI, with rank info."""
    word_lower = word.lower().strip()
    vocab_dict = get_vocab_dict()
    rank = vocab_dict.get(word_lower, 99999)

    client = get_openai_client()
    if not client:
        return {"error": "AI client not available"}

    model_name = get_config()["openai_model"]

    system_prompt = """# Role
Atomic Dictionary.

# Goal
Output ONE core meaning, ONE etymology, and TWO matching examples.

# Critical Constraint: ATOMIC SINGLE SENSE
- **Force Single Sense**: Regardless of how many meanings a word has, pick ONLY the #1 most common one.
- **Strict Alignment**: The Definition, Etymology, and BOTH Examples must strictly support this single meaning.
- **Format**: Follow the 4-line structure below perfectly.

# Output Format
[word] (lowercase)
[CN Meaning] | [Short EN Definition (<8 words)]
🌱 词源: [root (CN) + affix (CN)] (Explain origin briefly)
• [English Example 1] ([CN Trans])
• [English Example 2] ([CN Trans])

# Few-Shot Examples (Demonstrating Selection)
**User Input:**
spring

**Model Output:**
spring
春天 | The season after winter
🌱 词源: spring- (涌出/生长) → 万物复苏
• Flowers bloom in spring. (花朵在春天绽放。)
• I love the fresh air of spring. (我喜欢春天清新的空气。)

**User Input:**
date

**Model Output:**
date
日期 | Specific day of the month
🌱 词源: dat- (给予/指定) + -e (名词后缀)
• What is today's date? (今天是几号？)
• Please sign and date the form. (请在表格上签名并注明日期的。)

**User Input:**
express

**Model Output:**
express
表达；表示 | Convey a thought or feeling
🌱 词源: ex- (向外) + press (压/挤)
• She expressed her thanks to us. (她向我们表达了谢意。)
• Words cannot express my feelings. (言语无法表达我的感受。)"""

    user_prompt = word

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content
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
    client = get_openai_client()
    if not client:
        return None

    model_name = get_config()["openai_model"]
    total_words = len(words_list)
    full_results = []

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
                content = response.choices[0].message.content
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

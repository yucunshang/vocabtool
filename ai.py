# AI-backed word definitions and batch card generation.

import logging
import time
from typing import Any, Callable, Dict, List, Optional

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


def build_anki_prompt(
    words_list: List[str],
    card_template: str = constants.DEFAULT_CARD_TEMPLATE,
) -> str:
    """Build the third-party AI prompt for the selected Anki card template."""
    template = constants.CARD_TEMPLATES.get(
        card_template,
        constants.CARD_TEMPLATES[constants.DEFAULT_CARD_TEMPLATE],
    )
    words_block = "\n".join(words_list) if words_list else "[INSERT YOUR WORD LIST HERE]"

    template_instructions = {
        "word_front": """# Selected Card Template
Front: target word or phrase.
Back: Chinese meaning + English definition + English example + Chinese example.""",
        "example_front": """# Selected Card Template
Front: English example sentence with the target word/phrase bolded by Anki.
Back: Chinese meaning + English definition + English example + Chinese example.""",
        "definition_front": """# Selected Card Template
Front: English definition + part of speech + first-letter hint.
Back: target word/phrase + Chinese meaning + example sentence.""",
    }.get(card_template, """# Selected Card Template
Front: target word or phrase.
Back: Chinese meaning + English definition + English example + Chinese example.""")

    return f"""# Role
You are an expert English lexicographer and Anki card designer.
Create high-quality vocabulary cards for the selected template.
Process every input item exactly once.

{template_instructions}

# Input Data
{words_block}

# Output Format Guidelines
1. Output strictly inside one ```text code block.
2. One card per line.
3. Use `|||` as the only field delimiter.
4. Use exactly this 6-field structure:
   `Word/Phrase` ||| `Part of Speech` ||| `Chinese Meaning` ||| `English Definition` ||| `English Example` ||| `Chinese Example`

# Field Rules
1. Field 1: Use the exact target word unless the input itself is a phrase.
2. Field 2: Use concise part-of-speech labels, such as `noun`, `verb`, `adjective`, `adverb`, or `phrasal verb`.
3. Field 3: Simplified Chinese only. Keep it concise and aligned to the most common meaning.
4. Field 4: Short English definition, B1-C1 level, under 12 words.
5. Field 5: Natural English example sentence. It must contain the exact target word/phrase from Field 1.
6. Field 6: Natural Simplified Chinese translation of Field 5.
7. Do not output numbering, bullets, Markdown tables, explanations, or extra fields.
8. The app computes the first-letter hint automatically; do not add a hint field.

# Valid Example
```text
hectic ||| adjective ||| 忙乱的；忙碌的 ||| full of hurried activity ||| She had a hectic schedule all week. ||| 她整周日程都很忙乱。
altruism ||| noun ||| 利他主义；无私 ||| selfless concern for other people ||| Altruism motivated her anonymous donation. ||| 利他主义促使她匿名捐款。
```

# Task
Create cards using the `{template["label"]}` template and strictly follow the 6-field format."""


def process_ai_in_batches(
    words_list: List[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    card_template: str = constants.DEFAULT_CARD_TEMPLATE,
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
        user_prompt = build_anki_prompt(batch, card_template)

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

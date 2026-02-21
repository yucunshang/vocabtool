# =============================================================================
# AI 提示词模板集中存放 · 可直接修改下方内容，不影响制卡 / apkg / 语音 / 卡片格式
# =============================================================================

# -----------------------------------------------------------------------------
# ① 查词（Quick Lookup）系统提示词：单次查词时使用（6 行格式，3 条例句，不过度纠正）
# -----------------------------------------------------------------------------
LOOKUP_SYSTEM_PROMPT = """# Role
Atomic Flash Dictionary (Bilingual Edition) — Instant Lookup Mode

# Goal
Provide the #1 most common meaning of the input word/phrase in a strict 6-line format.
Target: Chinese learners. English definitions MUST use a restricted 2000-word defining vocabulary (similar to Longman/Oxford Learner's).

# 🔒 CORE RULES
1. **Single Sense Lock**: Select ONLY the primary meaning. Casing: china (瓷器) vs. China (中国).
2. **Minimalist Origin**: Explain etymology or logic in [Chinese] within 20 words. No academic jargon.
3. **Restricted English**: The English definition must be simple enough for a beginner. Avoid using the target word in its own definition.
4. **No Formatting**: RAW text only. Do NOT use Markdown (no **, no #). Use "•" for bullets.
5. **No Over-correction**: If the word exists (e.g., "stag"), do not change it (to "stage"). Correct ONLY obvious typos.
6. **Bilingual Examples**: 3 short, natural, A2-B2 level examples with Chinese translations.
7. **Fixed Order**: Process strictly in the 6-line template order. No extra chat.

# Output Template (Strictly 6 lines)
{term} ({pos} {CN_pos})
{CN_Meaning} | {Learner_English_Definition}
🌱 词源: {Simplified_Logic_in_CN}
• {Example_1} ({CN_Trans})
• {Example_2} ({CN_Trans})
• {Example_3} ({CN_Trans})

# Typo Handling
If a typo is detected, add one line ABOVE the template: ✔️ 拼写纠正: {typo} -> {correct}

# Few-Shot Example
User: compromise
AI:
compromise (n. 名词)
妥协 | an agreement where people reduce their demands to end an argument
🌱 词源: com (共同) + promise (承诺)，双方共同承诺退让。
• We reached a compromise after hours of talk. (经过数小时谈话，我们达成了妥协。)
• Life is full of compromise. (生活充满了折中与妥协。)
• Neither side was willing to compromise. (双方都不愿意让步。)

# Task
Identify the word/phrase, then output strictly following the template."""


# -----------------------------------------------------------------------------
# ② 批量制卡（Card Generation）内置 AI 专用
# 占位符：{words_str} 由 ai.build_card_prompt() 填入
# -----------------------------------------------------------------------------
CARD_GEN_SYSTEM_PROMPT = "You are a helpful assistant for vocabulary learning. Follow the required card format strictly and do not add extra commentary."

CARD_GEN_USER_TEMPLATE = """# Role
Atomic Flash Dictionary (Bilingual Edition) — Anki Card Batch Mode

# Goal
Convert a list of **words or phrases** into minimalist Anki cards. For each item, provide the **#1 most common meaning** in a strict 3-field format (one line per entry).
**Target Audience**: Chinese learners who need to grasp the meaning instantly.

# CORE RULES
1. **Single Sense Lock**: Select ONLY the primary meaning. Casing: china = porcelain (瓷器); China = country (中国).
2. **中英释义 (Bilingual Definition)**: [中文释义] | [Short English]. BOTH Chinese and English are REQUIRED. One core meaning only; keep both concise and natural.
3. **ONE Example**: Exactly one short, everyday English (A2-B1) example sentence with a natural Chinese translation in parentheses. The example MUST match the definition strictly.
4. **NO Etymology**: Do NOT output any etymology, roots, or affixes. Output only: word, definition, example.
5. **Phrase Support**: For phrases (e.g., "give up"), define the phrase as a unit.
6. **Format**: Output RAW text only. Absolutely NO Markdown, no code blocks, no backticks. One entry per line.
7. **Fixes**: Auto-capitalize proper nouns (e.g., english → English).
8. **Completeness**: Process every input item in original order; do not skip, merge, or reorder.
9. **Delimiter Safety**: Keep exactly 2 delimiters (`|||`) per line; never place `|||` inside field content.
10. **No Extra Text**: Output cards only, without headings, numbering, or explanations.

# Output Structure (Exactly 3 fields per line)
Separator: `|||`
Target Word ||| 中文释义 | Short English ||| English example. (中文翻译。)

# Valid Examples
spring ||| 春天 | The season after winter ||| Flowers bloom in spring. (花朵在春天绽放。)
give up ||| 放弃；戒除 | To stop doing something ||| I will never give up on my dreams. (我永远不会放弃我的梦想。)
date ||| 日期 | Specific day of the month ||| What is today's date? (今天是几号？)

# Input Data
{words_str}

# Task
Process the list (max 10 words per request). One line per word/phrase. 中英释义 (Chinese | English) REQUIRED, one example with natural Chinese translation, no etymology. Ensure zero errors."""

# -----------------------------------------------------------------------------
# ②b 阅读卡（语境填空）3 字段：挖空句 ||| 单词/音标 释义 ||| 例句
# -----------------------------------------------------------------------------
CARD_GEN_CLOZE_TEMPLATE = """# Role
Reading Card Generator. Create fill-in-the-blank sentences where ONLY the target word fits.

# Rules
1. **Uniqueness**: Add specific context so only one word fits. BAD: "He put it in the ___." GOOD: "He hung his coats in the wooden ___."
2. **Simplicity & Realism**: Use basic, everyday English (A2-B2) to construct realistic, natural adult scenarios (e.g., workplace, daily life). ONLY the target word should be difficult. Avoid childish or abstract sentences.
3. **Blank**: Use exactly eight underscores ________ where the word goes. No {{c1::}} or other markup.
4. **Meaning**: One line: word /IPA/ pos. 中文释义. Example: brass /bræs/ n. 黄铜
5. **Completeness**: Process every input item in original order; do not skip any item.
6. **Delimiter Safety**: Keep exactly 2 delimiters (`|||`) per line.

# Output Format
Exactly 3 fields per line, separated by ` ||| `. No markdown, no extra line breaks. Output raw text ONLY.
Field 1: Sentence with ________
Field 2: word /IPA/ pos. 中文释义
Field 3: Full sentence with answer. (中文翻译)

# Example
Input: brass, dam

Output:
The doorknob, made of polished ________, gleamed in the hallway light. ||| brass /bræs/ n. 黄铜 ||| The doorknob, made of polished brass, gleamed in the hallway light. (门把手在走廊灯光下闪闪发亮。)
The beavers built a wooden ________ across the stream to block the water. ||| dam /dæm/ n. 水坝 ||| The beavers built a wooden dam across the stream to block the water. (海狸在溪流上筑坝拦水。)

# Task
Process the following words (Max 10 per request). Output ONLY the cards, one per line:
{words_str}"""

# -----------------------------------------------------------------------------
# ②c 输出卡（表达方向：写作、口语）
# 正面=中文场景 反面=英文词块+例句
# -----------------------------------------------------------------------------
CARD_GEN_PRODUCTION_TEMPLATE = """# Role
Anki Card Designer for **Production Cards** (output direction: writing, speaking, active vocabulary).

# Goal
For each target word, output a Chinese scenario (what the learner wants to express) and the English chunk + example. Max 10 words per request.

# Output Format
Output RAW text only. Absolutely NO Markdown, no code blocks, no backticks. One entry per line. Separator: `|||`
`你想说：[自然流畅的中文场景描述]` ||| `[English chunk / collocation]` ||| `[English example]. ([中文翻译。])`

# Field Rules
1. **Field 1 (Front)**: Natural Chinese scenario—what the learner wants to say. It must sound like a real thought or conversation prompt. E.g., "你想说：这份声明措辞模糊，故意让人猜"
2. **Field 2 (Back/Target)**: The best English chunk or collocation for that scenario. CRITICAL: Do NOT just output the single target word. Output a natural 2-4 word phrase (e.g., target word + noun/verb/prep). E.g., "ambiguous statement".
3. **Field 3 (Back/Context)**: ONE short, everyday English (A2-B2) example sentence using the exact chunk from Field 2. Include a natural Chinese translation.
4. **Completeness**: Process every input item in original order; one line per item. Do not skip.
5. **Delimiter Safety**: Keep exactly 2 delimiters (`|||`) per line. Never use `|||` inside the text.

# Example
Input: ambiguous
Output:
你想说：这份声明措辞模糊，故意让人猜。 ||| ambiguous statement ||| The government's ambiguous statement left room for speculation. (政府含糊其辞的声明让人浮想联翩。)

# Input Data
{words_str}

# Task
Process the list (max 10 words). One scenario + chunk + example per word. Output RAW text only. Ensure zero errors."""

# -----------------------------------------------------------------------------
# ②d 中英互译卡（应试方向：初高中、中英互译题型）
# 正面=中文释义 反面=英文单词+音标+例句
# -----------------------------------------------------------------------------
CARD_GEN_TRANSLATION_TEMPLATE = """# Role
Anki Card Designer for **Translation Cards** (exam direction: C-E translation, active recall).

# Goal
For each target word, output the Chinese definition with a first-letter hint (front) and the English word + phonetic + example (back). Max 10 words per request.

# Output Format
Output RAW text only. Absolutely NO Markdown, no code blocks, no backticks. One entry per line. Separator: `|||`
`中文释义 (首字母...)` ||| `English word /IPA/` ||| `English example. (中文翻译。)`

# Field Rules
1. **Field 1 (Front)**: ONE concise Chinese definition. CRITICAL: You MUST add the first letter of the target English word followed by an ellipsis in parentheses to prevent synonym confusion. E.g. "模糊的，含混不清的 (a...)"
2. **Field 2 (Back)**: English word + IPA phonetic. E.g. "ambiguous /æmˈbɪɡjuəs/"
3. **Field 3 (Back/Context)**: ONE short, everyday English (A2-B2) example sentence using the target word. Include a natural Chinese translation.
4. **Completeness**: Process every input item in original order; one line per item. Do not skip.
5. **Delimiter Safety**: Keep exactly 2 delimiters (`|||`) per line. Never use `|||` inside the text.

# Example
Input: ambiguous
模糊的，含混不清的 (a...) ||| ambiguous /æmˈbɪɡjuəs/ ||| The instructions were ambiguous. (说明含糊不清。)

# Input Data
{words_str}

# Task
Process the list (max 10 words). One line per word. Output RAW text only. Ensure zero errors."""

# -----------------------------------------------------------------------------
# ②e 听音卡（先听发音再回忆）
# 正面=纯音频 反面=单词+释义+例句，格式同标准卡
# -----------------------------------------------------------------------------
CARD_GEN_AUDIO_TEMPLATE = """# Role
Anki Card Designer for **Audio Cards** (listen first, then recall).

# Goal
For each target word, output the same 3-field format as standard cards. The card shows audio on the front, then word + definition + example on the back.

# Output Format
Inside a single `text` code block. One entry per line. Separator: `|||`
`English word or phrase` ||| `中英释义` ||| `Example sentence. (中文翻译。)`

# Example
heavy rain ||| 大雨 | Intense rainfall ||| We got caught in heavy rain. (我们遇上了大雨。)

# Input Data
{words_str}

# Task
Process the list (max 10 words). Output ONLY the cards, one per line, inside ```text block."""

# -----------------------------------------------------------------------------
# ③ 第三方 AI 专用（可自定义格式，每批最多 200 词）
# 占位符由 ai.build_thirdparty_prompt(words_str, fmt) 填入
# -----------------------------------------------------------------------------
THIRD_PARTY_CARD_TEMPLATE = """# Role
You are a senior English Lexicographer and Anki Card Designer.
Your task is to convert the provided batch into high-quality, import-ready cards for serious learners.
This prompt is designed for strong frontier models. Prioritize semantic precision, natural usage, and consistency.

# Input Data
{words_str}

# Output Format Guidelines
1. **Output Container**: Strictly inside a single ```text code block.
2. **Layout**: One entry per line.
3. **Separator**: Use `|||` as the delimiter.
4. **Target Structure**:
   {structure_line}
5. **Completeness**: Process every input item in original order; no skipping, no merging.
6. **Delimiter Safety**: Do NOT use `|||` inside any field content.
7. **No Chat Noise**: Output cards only. No explanation, no analysis, no numbering.

# Field Constraints (Strict)
{field1_instruction}
{field2_instruction}
{field3_instruction}
{field4_instruction}

# Quality Bar (Strong-Model Requirements)
1. Choose the primary/common sense unless a phrase strongly implies another sense.
2. Definitions must be accurate, concise, and level-appropriate to the selected mode.
3. Example sentences must be natural, specific, and strictly aligned with the definition.
4. Avoid generic, robotic, or templated wording.
5. Maintain stable style and punctuation across all lines.

# Valid Example
{example_line}

# Task
Process the provided input list strictly adhering to the format above."""


# -----------------------------------------------------------------------------
# ③b 第三方 AI 固定卡型模板（强模型版，无总量限制；当前输入为单批，最多 200）
# -----------------------------------------------------------------------------
THIRD_PARTY_CLOZE_TEMPLATE = """# Role
Expert Lexicographer & Reading Card Generator for Strong LLMs.

# Input Data
{words_str}

# Global Rules
1. Exhaustive Processing: Process 100% of the items in this batch. Zero omissions are allowed.
2. Structure: Exactly one card per line. Exactly 3 fields per line, separated by the strict delimiter `|||`.
3. Delimiter Isolation: Never use the sequence `|||` within the content of any field.
4. Output Format: Output strictly inside a single ```text code block. Absolutely no introductory, conversational, or concluding text.

# Field Format
Field 1: Context-rich prompt sentence containing precise semantic clues. Replace the target word with exactly 8 underscores: `________`.
Field 2: Target word /IPA/ pos. 中文释义 (Strictly use {ipa_style} for IPA).
Field 3: Complete sentence restoring the target word. (Contextual, accurate Chinese translation).

# Quality Constraints
1. Absolute Cloze Uniqueness (Contextual Lockdown): The context in Field 1 MUST strictly isolate the target word using explicit contrasts, built-in definitions, or extreme consequences. If a generic word perfectly fits, the context is too weak.
2. Accessible yet Authentic Carrier Language (CRITICAL BALANCE): The sentence MUST be written in simple, highly readable English (A2-B2 level vocabulary) so the user can easily deduce the blank. However, "simple" does NOT mean childish or robotic. You must use these basic words to construct highly natural, idiomatic, and realistic adult scenarios (e.g., workplace, daily life, news events).
3. Linguistic Quality: Sentences must be grammatically flawless and concrete. Avoid generic names (e.g., "Person A", "Bob") or overly abstract concepts. Ground the sentence in a vivid, real-world situation.
4. Perfect Resolution: Field 3 must perfectly resolve Field 1, mirroring it exactly but with the target word replacing the blank.

# Example
Unlike his brother who usually made quick and careless choices, Arthur was so ________ that he spent three weeks checking every line of the apartment lease before finally signing it. ||| meticulous /məˈtɪkjələs/ adj. 一丝不苟的，缜密的 ||| Unlike his brother who usually made quick and careless choices, Arthur was so meticulous that he spent three weeks checking every line of the apartment lease before finally signing it. (与他那做选择通常既快又粗心的兄弟不同，亚瑟非常缜密，在最终签署公寓租赁合同前花了三个星期检查每一行条款。)

# Task
Generate the cards strictly based on the provided input and rules."""


THIRD_PARTY_TRANSLATION_TEMPLATE = """# Role
Translation Card Generator for strong LLMs.

# Input Data
{words_str}

# Global Rules
1. Process every input item in order.
2. One line per item. Exactly 3 fields separated by `|||`.
3. Do not use `|||` inside fields.
4. Output strictly inside one ```text code block. No explanation text.

# Field Format
`中文释义` ||| `English word / IPA` ||| `Example sentence. (中文翻译。)`

# Quality Constraints
1. Field 1 should be concise but discriminative Chinese meaning.
2. Field 2 should be canonical headword + clean IPA.
3. Field 3 should be natural, high-quality usage example.

# Example
模糊的，含混不清的 ||| ambiguous / æmˈbɪɡjuəs ||| The instructions were ambiguous. (说明含糊不清。)

# Task
Generate cards only."""


THIRD_PARTY_PRODUCTION_TEMPLATE = """# Role
Production Card Generator for strong LLMs (active output training).

# Input Data
{words_str}

# Global Rules
1. Process every input item in order.
2. One line per item. Exactly 3 fields separated by `|||`.
3. Do not use `|||` inside fields.
4. Output strictly inside one ```text code block. No explanation text.

# Field Format
`中文场景描述（你想说...）` ||| `English chunk / collocation` ||| `Example sentence. (中文翻译。)`

# Quality Constraints
1. Field 1 should be realistic communication intent in natural Chinese.
2. Field 2 should be the strongest, most usable chunk/collocation.
3. Field 3 must demonstrate Field 2 in a natural context.

# Example
你想说：这份声明措辞模糊，故意让人猜。 ||| ambiguous statement ||| The government's ambiguous statement left room for speculation. (政府含糊其辞的声明让人浮想联翩。)

# Task
Generate cards only."""


THIRD_PARTY_AUDIO_TEMPLATE = """# Role
Audio Card Generator for strong LLMs.

# Input Data
{words_str}

# Global Rules
1. Process every input item in order.
2. One line per item. Exactly 3 fields separated by `|||`.
3. Do not use `|||` inside fields.
4. Output strictly inside one ```text code block. No explanation text.

# Field Format
`English word or phrase` ||| `中英释义` ||| `Example sentence. (中文翻译。)`

# Quality Constraints
1. Keep Field 1 as clean headword/phrase only.
2. Field 2 must be balanced bilingual meaning (中文 | English).
3. Field 3 should be natural, specific, and easy to remember.

# Example
heavy rain ||| 大雨 | Intense rainfall ||| We got caught in heavy rain. (我们遇上了大雨。)

# Task
Generate cards only."""

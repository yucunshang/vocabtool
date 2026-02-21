# Constants and configuration for Vocab Flow Ultra.

DEFAULT_UPLOADER_ID = "1000"
APP_RELEASE_CHANNEL = "stable-anki-instant-dict"
MIN_RANDOM_ID = 100000
MAX_RANDOM_ID = 999999
REQUEST_TIMEOUT_SECONDS = 15
MAX_PREVIEW_CARDS = 10
BEIJING_TIMEZONE_OFFSET = 8
MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
QUICK_LOOKUP_COOLDOWN_SECONDS = 0
QUICK_LOOKUP_CACHE_MAX = 100

# Temp .apkg files: subdir under system temp, cleanup files older than this
APKG_TEMP_SUBDIR = "vocabflow_apkg"
APKG_CLEANUP_MAX_AGE_SECONDS = 24 * 3600  # 24 hours

MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 25

AI_BATCH_SIZE = 10     # 每组 10 词
AI_CONCURRENCY = 5     # 并发 5
MAX_AUTO_LIMIT = 500   # 一次性制卡上限
MAX_RETRIES = 2        # 失败重试 1 次，减少无效消耗
AI_BATCH_MAX_RETRIES = 4  # 批量制卡每组最多尝试 4 次（3 次重试），应对限流/超时

# Third-party prompt batching: unlimited total, split into chunks of this size.
THIRD_PARTY_PROMPT_BATCH_SIZE = 200

TTS_CONCURRENCY = 5
TTS_RETRY_ATTEMPTS = 3
MIN_AUDIO_FILE_SIZE = 100

# OpenAI-compatible API defaults (single source of truth for config.py / UI label)
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "deepseek-chat"
DEFAULT_OPENAI_MODEL_DISPLAY = "DeepSeek"

ANKI_MODEL_ID = 1842957301
ANKI_MODEL_CLOZE_ID = 1842957302
ANKI_MODEL_TRANSLATION_ID = 1842957303
ANKI_MODEL_PRODUCTION_ID = 1842957304
ANKI_MODEL_AUDIO_ID = 1842957305

# 卡片类型：standard=标准卡, cloze=阅读卡, translation=互译卡, production=表达卡, audio=听音卡
CARD_TYPES = ["standard", "cloze", "translation", "production", "audio"]

# 向后兼容：旧代码仍可读取该常量作为显示名 fallback
AI_MODEL_DISPLAY = DEFAULT_OPENAI_MODEL_DISPLAY

ENCODING_PRIORITY = ['utf-8', 'gb18030', 'latin-1']

DEFAULT_SESSION_STATE = {
    'uploader_id': DEFAULT_UPLOADER_ID,
    'anki_input_text': "",
    'anki_pkg_name': "",
    'quick_lookup_last_query': "",
    'quick_lookup_last_result': None,
    'quick_lookup_is_loading': False,
    'quick_lookup_block_until': 0.0,
    'quick_lookup_cache_keys': [],
    'extract_rank_preset': '常用 (6001–10000)',
    'extract_min_rank': 6001,
    'extract_max_rank': 10000,
}

# ---- Rate limiting (generous – designed to stop bots, not humans) ----
# AI word lookup
RL_LOOKUP_PER_MINUTE = 60       # ~1 word per second sustained
RL_LOOKUP_PER_HOUR = 500        # ~8 words/min sustained
RL_LOOKUP_PER_DAY = 3000        # heavy study day

# Batch AI card generation (each click = 1 event, not per-word)
RL_BATCH_PER_MINUTE = 5         # can't realistically click faster
RL_BATCH_PER_HOUR = 30          # generous for iterating
RL_BATCH_PER_DAY = 100          # very heavy usage day

# URL scraping
RL_URL_PER_MINUTE = 15          # pasting multiple articles
RL_URL_PER_HOUR = 150
RL_URL_PER_DAY = 500

# Max input length guards
MAX_LOOKUP_INPUT_LENGTH = 100   # single word/phrase lookup
MAX_PASTE_TEXT_LENGTH = 500_000 # ~500 KB of text
MAX_URL_LENGTH = 2048

# PDF: limit pages to keep extraction fast (text analysis has no character limit)
PDF_MAX_PAGES = 50              # only extract first N pages from PDF

# 词汇量区间预设（筛选单词通用，词表模式不适用）
RANK_PRESETS = [
    ("核心", 1, 2809),
    ("基础", 2810, 6000),
    ("常用", 6001, 10000),
    ("进阶", 10001, 15000),
    ("高级", 15001, 20000),
    ("专业", 20001, 50000),
]

VOICE_MAP = {
    "👩 美音女声 (Jenny)": "en-US-JennyNeural",
    "👨 美音男声 (Christopher)": "en-US-ChristopherNeural",
    "👩 英音女声 (Sonia)": "en-GB-SoniaNeural",
    "👨 英音男声 (Ryan)": "en-GB-RyanNeural",
}

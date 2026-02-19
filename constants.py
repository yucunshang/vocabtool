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

AI_BATCH_SIZE = 20   # 每批请求词数，越大 API 调用越少、整体越快，但单次响应更慢
MAX_AUTO_LIMIT = 200  # 内置 AI 一次性制卡上限
MAX_RETRIES = 3

TTS_CONCURRENCY = 4
TTS_RETRY_ATTEMPTS = 3
MIN_AUDIO_FILE_SIZE = 100

ANKI_MODEL_ID = 1842957301

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
    'extract_rank_preset': '中级 (6001–10000)',
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
    ("初学者", 1, 3000),
    ("初级", 3001, 6000),
    ("中级", 6001, 10000),
    ("高级", 10001, 15000),
    ("专业", 15001, 21000),
]

VOICE_MAP = {
    "👩 美音女声 (Jenny)": "en-US-JennyNeural",
    "👨 美音男声 (Christopher)": "en-US-ChristopherNeural",
    "👩 英音女声 (Sonia)": "en-GB-SoniaNeural",
    "👨 英音男声 (Ryan)": "en-GB-RyanNeural",
}


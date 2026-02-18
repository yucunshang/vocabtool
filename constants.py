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
QUICK_LOOKUP_COOLDOWN_SECONDS = 0.5
QUICK_LOOKUP_CACHE_MAX = 100

# Temp .apkg files: subdir under system temp, cleanup files older than this
APKG_TEMP_SUBDIR = "vocabflow_apkg"
APKG_CLEANUP_MAX_AGE_SECONDS = 24 * 3600  # 24 hours

MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 25

AI_BATCH_SIZE = 20   # 每批请求词数，越大 API 调用越少、整体越快，但单次响应更慢
MAX_AUTO_LIMIT = 200  # 内置 AI 一次性制卡上限
MAX_RETRIES = 3

TTS_CONCURRENCY = 3
TTS_RETRY_ATTEMPTS = 3
MIN_AUDIO_FILE_SIZE = 100

ANKI_MODEL_ID = 1842957301

ENCODING_PRIORITY = ['utf-8', 'gb18030', 'latin-1']

DEFAULT_SESSION_STATE = {
    'uploader_id': DEFAULT_UPLOADER_ID,
    'anki_input_text': "",
    'anki_pkg_name': "",
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

VOICE_MAP = {
    "👩 美音女声 (Jenny)": "en-US-JennyNeural",
    "👨 美音男声 (Christopher)": "en-US-ChristopherNeural",
    "👩 英音女声 (Sonia)": "en-GB-SoniaNeural",
    "👨 英音男声 (Ryan)": "en-GB-RyanNeural",
}


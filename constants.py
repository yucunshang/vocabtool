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
QUICK_LOOKUP_COOLDOWN_SECONDS = 1.2
QUICK_LOOKUP_CACHE_MAX = 100

# Temp .apkg files: subdir under system temp, cleanup files older than this
APKG_TEMP_SUBDIR = "vocabflow_apkg"
APKG_CLEANUP_MAX_AGE_SECONDS = 24 * 3600  # 24 hours

MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 25

AI_BATCH_SIZE = 10
MAX_AUTO_LIMIT = 300
MAX_RETRIES = 3

TTS_CONCURRENCY = 3
TTS_RETRY_ATTEMPTS = 3
MIN_AUDIO_FILE_SIZE = 100

ANKI_MODEL_ID_BASE = 1842957400

DEFAULT_CARD_TEMPLATE = "word_front"
CARD_TEMPLATES = {
    "word_front": {
        "label": "1. 正面单词",
        "description": "正面显示单词；反面显示中文释义、英文释义、英文例句、中文例句。",
    },
    "example_front": {
        "label": "2. 正面例句",
        "description": "正面显示例句并加粗目标词；反面显示中文释义、英文释义、英文例句、中文例句。",
    },
    "definition_front": {
        "label": "3. 英文释义 + 词性 + 首字母提示",
        "description": "正面显示英文释义、词性和首字母提示；反面显示单词、中文释义和例句。",
    },
}

ENCODING_PRIORITY = ['utf-8', 'gb18030', 'latin-1']

DEFAULT_SESSION_STATE = {
    'uploader_id': DEFAULT_UPLOADER_ID,
    'anki_input_text': "",
    'anki_pkg_name': "",
    'url_input_key': "",
}

VOICE_MAP = {
    "👩 美音女声 (Jenny)": "en-US-JennyNeural",
    "👨 美音男声 (Christopher)": "en-US-ChristopherNeural"
}

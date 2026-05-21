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

AI_BATCH_SIZE = 20
MAX_AUTO_LIMIT = 1000
MAX_PROMPT_BATCH_SIZE = 1000
MAX_RETRIES = 3

DIRECT_WORD_DEFAULT_COUNT = 200
DIRECT_WORD_MAX_COUNT = 5000
DEFAULT_DIRECT_WORD_PRIORITY = "input_order"
DIRECT_WORD_PRIORITY_OPTIONS = {
    "input_order": {
        "label": "按输入顺序",
        "description": "保留原始列表顺序，适合你已经排好优先级的单词表。",
    },
    "rank_ascending": {
        "label": "基础优先 (Rank 小→大)",
        "description": "优先保留内置词典中排名更靠前、使用频率更高的单词。",
    },
    "rank_descending": {
        "label": "进阶优先 (Rank 大→小)",
        "description": "优先保留排名更靠后、更进阶的词；词典外单词排在最后。",
    },
    "unknown_first": {
        "label": "未知词优先",
        "description": "优先保留内置词典未收录的词，再按进阶优先排序。",
    },
}

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

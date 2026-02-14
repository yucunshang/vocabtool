"""
Vocab Flow Ultra - Refactored Version
A Streamlit app for vocabulary learning with Anki integration and TTS features.
"""

import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
import zlib
import sqlite3
import asyncio
import edge_tts
import requests
import shutil
import zipfile
import tempfile
import traceback
import logging
from collections import Counter
from io import StringIO
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Protocol, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("OpenAI library not available")

# ==========================================
# Constants
# ==========================================
DEFAULT_UPLOADER_ID = "1000"
MIN_RANDOM_ID = 100000
MAX_RANDOM_ID = 999999
REQUEST_TIMEOUT_SECONDS = 10
MAX_PREVIEW_CARDS = 10
BEIJING_TIMEZONE_OFFSET = 8

# Word validation constants
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 25

# AI processing constants
AI_BATCH_SIZE = 10
MAX_AUTO_LIMIT = 300
MAX_RETRIES = 3

# TTS constants
TTS_CONCURRENCY = 3
TTS_RETRY_ATTEMPTS = 3
MIN_AUDIO_FILE_SIZE = 100

# Anki model constants
ANKI_MODEL_ID = 1842957301

# File encoding fallbacks
ENCODING_PRIORITY = ['utf-8', 'gb18030', 'latin-1']

# Session state default values
DEFAULT_SESSION_STATE = {
    'uploader_id': DEFAULT_UPLOADER_ID,
    'anki_input_text': "",
    'anki_pkg_data': None,
    'anki_pkg_name': "",
    'txt_pkg_data': None,
    'txt_pkg_name': "",
    'url_input_key': "",
}

# Voice mapping
VOICE_MAP = {
    "👩 美音女声 (Jenny)": "en-US-JennyNeural",
    "👨 美音男声 (Christopher)": "en-US-ChristopherNeural"
}

# ==========================================
# Protocols and Type Definitions
# ==========================================
class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""
    def __call__(self, ratio: float, message: str) -> None:
        ...

# ==========================================
# Error Handler
# ==========================================
class ErrorHandler:
    """Centralized error handling for consistent user feedback."""
    
    @staticmethod
    def handle(error: Exception, context: str, show_user: bool = True) -> None:
        """Handle errors consistently with logging and user feedback."""
        logger.error(f"{context}: {error}", exc_info=True)
        
        if show_user:
            st.error(f"❌ {context}: {str(error)}")
    
    @staticmethod
    def handle_with_fallback(error: Exception, fallback_value: Any, context: str = "") -> Any:
        """Handle error and return fallback value."""
        logger.warning(f"Error in {context}: {error}")
        return fallback_value
    
    @staticmethod
    def handle_file_error(error: Exception, file_type: str) -> str:
        """Handle file processing errors."""
        error_msg = f"Error processing {file_type}: {error}"
        logger.error(error_msg)
        return error_msg

# ==========================================
# Utility Functions
# ==========================================
def safe_str_clean(value: Any) -> str:
    """Convert to string and clean whitespace safely."""
    if value is None:
        return ""
    return str(value).strip()

def get_beijing_time_str() -> str:
    """Get current Beijing time as formatted string."""
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=BEIJING_TIMEZONE_OFFSET)
    return beijing_now.strftime('%m%d_%H%M')

def detect_file_encoding(bytes_data: bytes) -> str:
    """Detect file encoding using chardet if available, fallback to priority list."""
    try:
        import chardet
        detected = chardet.detect(bytes_data)
        encoding = detected.get('encoding')
        if encoding and detected.get('confidence', 0) > 0.7:
            return encoding
    except ImportError:
        logger.debug("chardet not available, using fallback encodings")
    
    # Fallback to trying encodings in priority order
    for encoding in ENCODING_PRIORITY:
        try:
            bytes_data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    
    return 'latin-1'  # Last resort

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra",
    page_icon="⚡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Custom CSS
st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .ai-warning { font-size: 12px; color: #666; margin-top: -5px; margin-bottom: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Resource Loading (Cached)
# ==========================================
@st.cache_resource(show_spinner="正在加载 NLP 引擎...")
def load_nlp_resources() -> Tuple[Any, Any]:
    """Load NLTK and lemminflect resources with proper error handling."""
    import nltk
    import lemminflect
    
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        required_packages = ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']
        for pkg in required_packages:
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {pkg}")
                nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        ErrorHandler.handle(e, "NLP 资源加载失败")
    
    return nltk, lemminflect

@st.cache_resource
def get_file_parsers() -> Tuple[Any, Any, Any, Any, Any]:
    """Lazy load file parsing libraries (cached)."""
    import pypdf
    import docx
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    return pypdf, docx, ebooklib, epub, BeautifulSoup

@st.cache_resource
def get_genanki() -> Tuple[Any, Any]:
    """Lazy load genanki library (cached)."""
    import genanki
    import tempfile
    return genanki, tempfile

@st.cache_data
def load_vocab_data() -> Tuple[Dict[str, int], Optional[pd.DataFrame]]:
    """Load vocabulary data from pickle or CSV files."""
    # Try pickle first
    if os.path.exists("vocab.pkl"):
        try:
            df = pd.read_pickle("vocab.pkl")
            vocab_dict = pd.Series(df['rank'].values, index=df['word']).to_dict()
            return vocab_dict, df
        except (FileNotFoundError, pd.errors.PickleError, KeyError) as e:
            logger.warning(f"Could not load pickle file: {e}")

    # Fallback to CSV files
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            
            word_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            rank_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
            
            df = df.dropna(subset=[word_col])
            df[word_col] = df[word_col].astype(str).str.lower().str.strip()
            df[rank_col] = pd.to_numeric(df[rank_col], errors='coerce')
            df = df.sort_values(rank_col).drop_duplicates(subset=[word_col], keep='first')
            
            vocab_dict = pd.Series(df[rank_col].values, index=df[word_col]).to_dict()
            return vocab_dict, df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return {}, None
    
    logger.warning("No vocabulary data files found")
    return {}, None

VOCAB_DICT, FULL_DF = load_vocab_data()

# ==========================================
# State Management
# ==========================================
def clear_all_state() -> None:
    """Clear all session state for fresh start."""
    if 'url_input_key' in st.session_state:
        st.session_state['url_input_key'] = ""
    
    keys_to_drop = [
        'gen_words_data', 'raw_count', 'process_time', 'stats_info',
        'anki_pkg_data', 'anki_pkg_name', 'anki_input_text',
        'txt_pkg_data', 'txt_pkg_name'
    ]
    
    for key in keys_to_drop:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state['uploader_id'] = str(random.randint(MIN_RANDOM_ID, MAX_RANDOM_ID))
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

# ==========================================
# Text Extraction Functions
# ==========================================
def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL."""
    _, _, _, _, BeautifulSoup = get_file_parsers()
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
            element.decompose()
        
        return soup.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        return ErrorHandler.handle_file_error(e, "URL")
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "URL parsing")

def extract_from_txt(uploaded_file: Any) -> str:
    """Extract text from TXT file."""
    bytes_data = uploaded_file.getvalue()
    encoding = detect_file_encoding(bytes_data)
    
    try:
        return bytes_data.decode(encoding)
    except UnicodeDecodeError as e:
        logger.warning(f"Decode failed with {encoding}, trying latin-1")
        return bytes_data.decode('latin-1', errors='ignore')

def extract_from_pdf(uploaded_file: Any) -> str:
    """Extract text from PDF file."""
    pypdf, _, _, _, _ = get_file_parsers()
    
    try:
        reader = pypdf.PdfReader(uploaded_file)
        pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(pages_text)
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "PDF")

def extract_from_docx(uploaded_file: Any) -> str:
    """Extract text from DOCX file."""
    _, docx, _, _, _ = get_file_parsers()
    
    try:
        doc = docx.Document(uploaded_file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "DOCX")

def extract_from_epub(uploaded_file: Any) -> str:
    """Extract text from EPUB file."""
    _, _, ebooklib, epub, BeautifulSoup = get_file_parsers()
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        book = epub.read_epub(tmp_path)
        text_parts = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text_parts.append(soup.get_text(separator=' ', strip=True))
        
        return " ".join(text_parts)
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "EPUB")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning(f"Could not remove temp file: {e}")

def extract_from_sqlite(uploaded_file: Any) -> str:
    """Extract text from SQLite database file."""
    tmp_db_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_db:
            tmp_db.write(uploaded_file.getvalue())
            tmp_db_path = tmp_db.name
        
        # Use context manager for database connection
        with sqlite3.connect(tmp_db_path) as conn:
            cursor = conn.cursor()
            
            # Try stem column first
            try:
                cursor.execute("SELECT stem FROM WORDS WHERE stem IS NOT NULL")
                rows = cursor.fetchall()
                text = " ".join([row[0] for row in rows if row[0]])
                
                if not text:
                    # Fallback to word column
                    cursor.execute("SELECT word FROM WORDS")
                    rows = cursor.fetchall()
                    text = " ".join([row[0] for row in rows if row[0]])
                
                return text
            except sqlite3.OperationalError as e:
                return f"Error reading DB schema: {e}"
    
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "SQLite DB")
    finally:
        if tmp_db_path and os.path.exists(tmp_db_path):
            try:
                os.remove(tmp_db_path)
            except OSError as e:
                logger.warning(f"Could not remove temp DB: {e}")

def extract_text_from_file(uploaded_file: Any) -> str:
    """Main function to extract text from uploaded files."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    extractors = {
        'txt': extract_from_txt,
        'pdf': extract_from_pdf,
        'docx': extract_from_docx,
        'epub': extract_from_epub,
        'db': extract_from_sqlite,
        'sqlite': extract_from_sqlite,
    }
    
    extractor = extractors.get(file_type)
    if extractor:
        return extractor(uploaded_file)
    
    return f"Unsupported file type: {file_type}"

# ==========================================
# Word Validation and Analysis
# ==========================================
def is_valid_word(word: str) -> bool:
    """Validate if a word meets criteria for processing."""
    if len(word) < MIN_WORD_LENGTH or len(word) > MAX_WORD_LENGTH:
        return False
    if re.search(r'(.)\1{2,}', word):  # Repeated characters
        return False
    if not re.search(r'[aeiouy]', word):  # Must have vowel
        return False
    return True

def get_lemma(word: str, lemminflect: Any) -> str:
    """Get lemma of a word with error handling."""
    try:
        lemmas = lemminflect.getLemma(word, upos='VERB')
        return lemmas[0] if lemmas else word
    except Exception:
        return word

def analyze_logic(
    text: str,
    current_level: int,
    target_level: int,
    include_unknown: bool
) -> Tuple[List[Tuple[str, int]], int, Dict[str, float]]:
    """Analyze text to extract vocabulary within specified rank range."""
    nltk, lemminflect = load_nlp_resources()
    
    # Extract tokens
    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_raw_count = len(raw_tokens)
    
    # Filter and count valid tokens
    valid_tokens = [token.lower() for token in raw_tokens if is_valid_word(token.lower())]
    token_counts = Counter(valid_tokens)
    
    # Statistics tracking
    stats_known_count = 0
    stats_target_count = 0
    stats_valid_total = sum(token_counts.values())
    
    # Process candidates
    final_candidates = []
    seen_lemmas = set()
    
    for word, count in token_counts.items():
        lemma = get_lemma(word, lemminflect)
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(word, 99999)
        
        # Determine best rank
        if rank_lemma != 99999 and rank_orig != 99999:
            best_rank = min(rank_lemma, rank_orig)
        elif rank_lemma != 99999:
            best_rank = rank_lemma
        else:
            best_rank = rank_orig
        
        # Update statistics
        if best_rank < current_level:
            stats_known_count += count
        elif current_level <= best_rank <= target_level:
            stats_target_count += count
        
        # Add to candidates if in range
        is_in_range = (best_rank >= current_level and best_rank <= target_level)
        is_unknown_included = (best_rank == 99999 and include_unknown)
        
        if is_in_range or is_unknown_included:
            word_to_keep = lemma if rank_lemma != 99999 else word
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)
    
    # Sort by rank
    final_candidates.sort(key=lambda x: x[1])
    
    # Calculate coverage statistics
    coverage_ratio = (stats_known_count / stats_valid_total) if stats_valid_total > 0 else 0
    target_ratio = (stats_target_count / stats_valid_total) if stats_valid_total > 0 else 0
    
    stats_info = {
        "coverage": coverage_ratio,
        "target_density": target_ratio
    }
    
    return final_candidates, total_raw_count, stats_info

# ==========================================
# AI Processing
# ==========================================
def get_openai_client() -> Optional[Any]:
    """Get configured OpenAI client with proper error handling."""
    if not OpenAI:
        st.error("❌ 未安装 OpenAI 库，无法使用内置 AI 功能。")
        return None
    
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ 未找到 OPENAI_API_KEY。请在 .streamlit/secrets.toml 中配置。")
        return None
    
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        ErrorHandler.handle(e, "Failed to initialize OpenAI client")
        return None

def process_ai_in_batches(
    words_list: List[str],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Optional[str]:
    """Process words in batches using AI with progress reporting."""
    client = get_openai_client()
    if not client:
        return None
    
    model_name = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    total_words = len(words_list)
    full_results = []
    
    system_prompt = "You are a helpful assistant for vocabulary learning."
    
    for i in range(0, total_words, AI_BATCH_SIZE):
        batch = words_list[i:i + AI_BATCH_SIZE]
        current_batch_str = "\n".join(batch)
        
        user_prompt = f"""# Role
You are an expert English Lexicographer.
# Input Data
{current_batch_str}

# Output Format Guidelines
1. **Output Container**: Strictly inside a single ```text code block.
2. **Layout**: One entry per line.
3. **Separator**: Use `|||` as the delimiter.
4. **Target Structure**:
   `Natural Phrase/Collocation` ||| `Concise Definition of the Phrase` ||| `Short Example Sentence` ||| `Etymology breakdown (Simplified Chinese)`

# Field Constraints
1. Field 1: Phrase - DO NOT output the single target word. Generate a high-frequency collocation.
2. Field 2: Definition - Define the *phrase* in English (B2-C1).
3. Field 3: Example - Authentic sentence.
4. Field 4: Etymology - Simplified Chinese.

# Valid Example
Input: hectic
Output:
a hectic schedule ||| a timeline full of frantic activity and very busy ||| She has a hectic schedule with meetings all day. ||| hect- (持续的) + -ic (形容词后缀)

# Task
Process the input list strictly."""
        
        for attempt in range(MAX_RETRIES):
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
                    processed_count = min(i + AI_BATCH_SIZE, total_words)
                    progress_callback(processed_count, total_words)
                
                break
            
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1 + attempt)
                    continue
                else:
                    ErrorHandler.handle(
                        e,
                        f"Batch {i//AI_BATCH_SIZE + 1} failed after {MAX_RETRIES} attempts",
                        show_user=True
                    )
    
    return "\n".join(full_results)

# ==========================================
# Anki Data Parsing
# ==========================================
def parse_anki_data(raw_text: str) -> List[Dict[str, str]]:
    """Parse AI-generated text into structured Anki card data."""
    parsed_cards = []
    text = raw_text.strip()
    
    # Extract code block if present
    code_block = re.search(r'```(?:text|csv)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1)
    else:
        text = re.sub(r'^```.*$', '', text, flags=re.MULTILINE)
    
    lines = text.split('\n')
    seen_phrases = set()
    
    for line in lines:
        line = line.strip()
        if not line or "|||" not in line:
            continue
        
        parts = line.split("|||")
        if len(parts) < 2:
            continue
        
        # Parse fields: Phrase ||| Definition ||| Example ||| Etymology
        phrase = parts[0].strip()
        meaning = parts[1].strip()
        example = parts[2].strip() if len(parts) > 2 else ""
        etymology = parts[3].strip() if len(parts) > 3 else ""
        
        # Skip duplicates
        if phrase.lower() in seen_phrases:
            continue
        seen_phrases.add(phrase.lower())
        
        parsed_cards.append({
            'w': phrase,
            'm': meaning,
            'e': example,
            'r': etymology
        })
    
    return parsed_cards

# ==========================================
# TTS Audio Generation
# ==========================================
async def _generate_audio_batch(
    tasks: List[Dict[str, str]],
    concurrency: int = TTS_CONCURRENCY,
    progress_callback: Optional[ProgressCallback] = None
) -> None:
    """Generate audio files concurrently with retry logic."""
    semaphore = asyncio.Semaphore(concurrency)
    total_files = len(tasks)
    completed_files = 0
    
    async def worker(task: Dict[str, str]) -> None:
        nonlocal completed_files
        async with semaphore:
            # Random jitter to simulate human behavior
            await asyncio.sleep(random.uniform(0.1, 0.8))
            
            success = False
            error_msg = ""
            
            for attempt in range(TTS_RETRY_ATTEMPTS):
                try:
                    if not os.path.exists(task['path']):
                        # Create new Communicate object for each retry
                        comm = edge_tts.Communicate(task['text'], task['voice'])
                        await comm.save(task['path'])
                        
                        # Verify file was created and is valid size
                        if os.path.exists(task['path']) and os.path.getsize(task['path']) > MIN_AUDIO_FILE_SIZE:
                            success = True
                            break
                        else:
                            if os.path.exists(task['path']):
                                os.remove(task['path'])
                            raise Exception("File size too small")
                    else:
                        success = True
                        break
                except Exception as e:
                    error_msg = str(e)
                    await asyncio.sleep(1.5 * (attempt + 1))
            
            if not success:
                logger.error(f"TTS failed for: {task['text']} | Error: {error_msg}")
            
            completed_files += 1
            if progress_callback:
                progress_callback(
                    completed_files / total_files,
                    f"正在生成 ({completed_files}/{total_files})"
                )
    
    jobs = [worker(task) for task in tasks]
    await asyncio.gather(*jobs, return_exceptions=True)

def run_async_batch(
    tasks: List[Dict[str, str]],
    concurrency: int = TTS_CONCURRENCY,
    progress_callback: Optional[ProgressCallback] = None
) -> None:
    """Run async audio generation batch."""
    if not tasks:
        return
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_generate_audio_batch(tasks, concurrency, progress_callback))
    finally:
        loop.close()

# ==========================================
# Anki Package Generation
# ==========================================
def generate_anki_package(
    cards_data: List[Dict[str, str]],
    deck_name: str,
    enable_tts: bool = False,
    tts_voice: str = "en-US-JennyNeural",
    progress_callback: Optional[ProgressCallback] = None
) -> str:
    """Generate Anki package (.apkg) file with optional TTS audio."""
    genanki, tempfile = get_genanki()
    media_files = []
    
    # CSS styling for cards
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .meaning { font-size: 20px; font-weight: bold; color: #222; margin-bottom: 15px; text-align: left; }
    .nightMode .meaning { color: #e0e0e0; }
    .example { 
        background: #f7f9fa; 
        padding: 15px; 
        border-left: 5px solid #0056b3; 
        border-radius: 4px; 
        color: #444; 
        font-style: italic; 
        font-size: 24px; 
        line-height: 1.5;
        text-align: left; 
        margin-bottom: 15px; 
    }
    .nightMode .example { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .etymology { display: block; font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; border: 1px solid #fef3c7; }
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
    """
    
    # Create model
    DECK_ID = zlib.adler32(deck_name.encode('utf-8'))
    
    model = genanki.Model(
        ANKI_MODEL_ID,
        'VocabFlow Unified Model',
        fields=[
            {'name': 'Phrase'}, {'name': 'Meaning'},
            {'name': 'Example'}, {'name': 'Etymology'},
            {'name': 'Audio_Phrase'}, {'name': 'Audio_Example'}
        ],
        templates=[{
            'name': 'Vocab Card',
            'qfmt': '''
                <div class="phrase">{{Phrase}}</div>
                <div>{{Audio_Phrase}}</div>
            ''',
            'afmt': '''
            {{FrontSide}}
            <hr>
            <div class="meaning">{{Meaning}}</div>
            <div class="example">🗣️ {{Example}}</div>
            <div>{{Audio_Example}}</div>
            {{#Etymology}}
            <div class="etymology">🌱 词源: {{Etymology}}</div>
            {{/Etymology}}
            ''',
        }], css=CSS
    )
    
    deck = genanki.Deck(DECK_ID, deck_name)
    tmp_dir = tempfile.gettempdir()
    
    notes_buffer = []
    audio_tasks = []
    
    # Prepare notes and audio tasks
    for idx, card in enumerate(cards_data):
        phrase = safe_str_clean(card.get('w', ''))
        meaning = safe_str_clean(card.get('m', ''))
        example = safe_str_clean(card.get('e', ''))
        etymology = safe_str_clean(card.get('r', ''))
        
        audio_phrase_field = ""
        audio_example_field = ""
        
        if enable_tts and phrase:
            safe_phrase = re.sub(r'[^a-zA-Z0-9]', '_', phrase)[:20]
            unique_id = int(time.time() * 1000) + random.randint(0, 9999)
            
            # Phrase audio
            phrase_filename = f"tts_{safe_phrase}_{unique_id}_p.mp3"
            phrase_path = os.path.join(tmp_dir, phrase_filename)
            audio_tasks.append({
                'text': phrase,
                'path': phrase_path,
                'voice': tts_voice
            })
            media_files.append(phrase_path)
            audio_phrase_field = f"[sound:{phrase_filename}]"
            
            # Example audio
            if example and len(example) > 3:
                example_filename = f"tts_{safe_phrase}_{unique_id}_e.mp3"
                example_path = os.path.join(tmp_dir, example_filename)
                audio_tasks.append({
                    'text': example,
                    'path': example_path,
                    'voice': tts_voice
                })
                media_files.append(example_path)
                audio_example_field = f"[sound:{example_filename}]"
        
        note = genanki.Note(
            model=model,
            fields=[phrase, meaning, example, etymology, audio_phrase_field, audio_example_field]
        )
        notes_buffer.append(note)
    
    # Generate audio files if needed
    if audio_tasks:
        def internal_progress(ratio: float, msg: str) -> None:
            if progress_callback:
                progress_callback(ratio, msg)
        
        run_async_batch(audio_tasks, concurrency=TTS_CONCURRENCY, progress_callback=internal_progress)
    
    # Add notes to deck
    for note in notes_buffer:
        deck.add_note(note)
    
    if progress_callback:
        progress_callback(1.0, "📦 正在打包 .apkg 文件...")
    
    # Create package
    package = genanki.Package(deck)
    package.media_files = media_files
    
    # Write to file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        package.write_to_file(tmp.name)
        
        # Clean up temporary audio files
        for file_path in media_files:
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Could not remove audio file {file_path}: {e}")
        
        return tmp.name

# ==========================================
# UI Components
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv` 或 `vocab.pkl` 文件，请检查目录。")

with st.expander("📖 使用指南 & 支持格式"):
    st.markdown("""
    **🚀 极速工作流**
    1. **提取**：支持 URL、PDF, ePub, Docx, txt 等格式。
    2. **生成**：自动完成文本生成、**并发语音合成**并打包下载。
    3. **优化**：支持导入 Anki 导出文本或 CSV，**自动添加语音**并打包。
    """)

tab_extract, tab_anki, tab_optimize = st.tabs([
    "1️⃣ 单词提取",
    "2️⃣ 卡片制作",
    "3️⃣ 文本转语音(TXT->Anki)"
])

# ==========================================
# Tab 1: Word Extraction
# ==========================================
with tab_extract:
    mode_context, mode_direct, mode_rank = st.tabs([
        "📄 语境分析",
        "📝 直接输入",
        "🔢 词频列表"
    ])
    
    with mode_context:
        col1, col2 = st.columns(2)
        current_rank = col1.number_input("忽略前 N 高频词 (Min Rank)", 1, 20000, 6000, step=100)
        target_rank = col2.number_input("忽略后 N 低频词 (Max Rank)", 2000, 50000, 10000, step=500)
        
        st.markdown("#### 📥 导入内容")
        
        input_url = st.text_input(
            "🔗 输入文章 URL (自动抓取)",
            placeholder="https://www.economist.com/...",
            key="url_input_key"
        )
        
        uploaded_file = st.file_uploader(
            "或直接上传文件",
            type=['txt', 'pdf', 'docx', 'epub', 'db', 'sqlite'],
            key=st.session_state['uploader_id'],
            label_visibility="collapsed"
        )
        
        pasted_text = st.text_area(
            "或在此粘贴文本",
            height=100,
            key="paste_key",
            placeholder="支持直接粘贴文章内容..."
        )
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("🔍 正在加载资源并分析文本...", expanded=True) as status:
                start_time = time.time()
                raw_text = ""
                
                if input_url:
                    status.write(f"🌐 正在抓取 URL: {input_url}...")
                    raw_text = extract_text_from_url(input_url)
                elif uploaded_file:
                    raw_text = extract_text_from_file(uploaded_file)
                else:
                    raw_text = pasted_text
                
                if len(raw_text) > 2:
                    status.write("🧠 正在进行 NLP 词形还原与分级...")
                    final_data, raw_count, stats_info = analyze_logic(
                        raw_text, current_rank, target_rank, False
                    )
                    
                    st.session_state['gen_words_data'] = final_data
                    st.session_state['raw_count'] = raw_count
                    st.session_state['stats_info'] = stats_info
                    st.session_state['process_time'] = time.time() - start_time
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容为空或太短", state="error")
    
    with mode_direct:
        raw_input = st.text_area(
            "✍️ 粘贴单词列表 (每行一个 或 逗号分隔)",
            height=200,
            placeholder="altruism\nhectic\nserendipity"
        )
        
        if st.button("🚀 生成列表", key="btn_direct", type="primary"):
            with st.spinner("正在解析列表..."):
                if raw_input.strip():
                    words = [w.strip() for w in re.split(r'[,\n\t]+', raw_input) if w.strip()]
                    unique_words = []
                    seen = set()
                    
                    for word in words:
                        if word.lower() not in seen:
                            seen.add(word.lower())
                            unique_words.append(word)
                    
                    data_list = [(w, VOCAB_DICT.get(w.lower(), 99999)) for w in unique_words]
                    st.session_state['gen_words_data'] = data_list
                    st.session_state['raw_count'] = len(unique_words)
                    st.session_state['stats_info'] = None
                    st.toast(f"✅ 已加载 {len(unique_words)} 个单词", icon="🎉")
                else:
                    st.warning("⚠️ 内容为空。")
    
    with mode_rank:
        gen_type = st.radio("生成模式", ["🔢 顺序生成", "🔀 随机抽取"], horizontal=True)
        
        if "顺序生成" in gen_type:
            col_a, col_b = st.columns(2)
            start_rank = col_a.number_input("起始排名", 1, 20000, 8000, step=100)
            count = col_b.number_input("数量", 10, 5000, 10, step=10)
            
            if st.button("🚀 生成列表"):
                with st.spinner("正在提取..."):
                    if FULL_DF is not None:
                        rank_col = next(c for c in FULL_DF.columns if 'rank' in c)
                        word_col = next(c for c in FULL_DF.columns if 'word' in c)
                        subset = FULL_DF[FULL_DF[rank_col] >= start_rank].sort_values(rank_col).head(count)
                        st.session_state['gen_words_data'] = list(zip(subset[word_col], subset[rank_col]))
                        st.session_state['raw_count'] = 0
                        st.session_state['stats_info'] = None
        else:
            col_min, col_max, col_cnt = st.columns([1, 1, 1])
            min_rank = col_min.number_input("最小排名", 1, 20000, 12000, step=100)
            max_rank = col_max.number_input("最大排名", 1, 25000, 15000, step=100)
            random_count = col_cnt.number_input("抽取数量", 10, 5000, 10, step=10)
            
            if st.button("🎲 随机抽取"):
                with st.spinner("正在抽取..."):
                    if FULL_DF is not None:
                        rank_col = next(c for c in FULL_DF.columns if 'rank' in c)
                        word_col = next(c for c in FULL_DF.columns if 'word' in c)
                        mask = (FULL_DF[rank_col] >= min_rank) & (FULL_DF[rank_col] <= max_rank)
                        candidates = FULL_DF[mask]
                        
                        if len(candidates) > 0:
                            subset = candidates.sample(n=min(random_count, len(candidates))).sort_values(rank_col)
                            st.session_state['gen_words_data'] = list(zip(subset[word_col], subset[rank_col]))
                            st.session_state['raw_count'] = 0
                            st.session_state['stats_info'] = None
    
    if st.button("🗑️ 清空重置", type="secondary", on_click=clear_all_state, key="btn_clear_extract"):
        pass
    
    # Display results if available
    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data_pairs = st.session_state['gen_words_data']
        words_only = [pair[0] for pair in data_pairs]
        
        st.divider()
        st.markdown("### 📊 分析报告")
        
        key1, key2, key3, key4 = st.columns(4)
        raw_count = st.session_state.get('raw_count', 0)
        stats = st.session_state.get('stats_info', {})
        
        key1.metric("总词数", f"{raw_count:,}")
        key2.metric("熟词覆盖", f"{stats.get('coverage', 0):.1%}" if stats else "--")
        key3.metric("生词密度", f"{stats.get('target_density', 0):.1%}" if stats else "--")
        key4.metric("提取生词", f"{len(words_only)}")
        
        display_text = ", ".join(words_only)
        with st.expander("📋 预览所有单词", expanded=False):
            st.code(display_text, language="text")
        
        st.divider()
        st.subheader("🤖 一键生成 Anki 牌组")
        
        st.write("🎙️ **语音设置**")
        
        selected_voice_label = st.radio(
            "选择发音人",
            options=list(VOICE_MAP.keys()),
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="extract_voice_radio"
        )
        selected_voice_code = VOICE_MAP[selected_voice_label]
        
        st.write("")
        enable_audio_auto = st.checkbox("✅ 启用 TTS 语音生成", value=True, key="chk_audio_auto")
        
        st.write("")
        
        col_ai_btn, col_copy_hint = st.columns([1, 2])
        
        with col_ai_btn:
            if st.button("✨ 使用 DeepSeek 生成", type="primary", use_container_width=True):
                target_words = words_only[:MAX_AUTO_LIMIT]
                
                if len(words_only) > MAX_AUTO_LIMIT:
                    st.warning(f"⚠️ 单词过多，自动截取前 {MAX_AUTO_LIMIT} 个进行处理。")
                
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                def update_ai_progress(current: int, total: int) -> None:
                    percent = current / total
                    progress_bar.progress(percent)
                    status_text.markdown(f"🤖 **DeepSeek 思考中...** ({current}/{total})")
                
                with st.spinner("🤖 DeepSeek 正在生成内容..."):
                    ai_result = process_ai_in_batches(target_words, progress_callback=update_ai_progress)
                
                if ai_result:
                    st.session_state['anki_input_text'] = ai_result
                    
                    parsed_data = parse_anki_data(ai_result)
                    if parsed_data:
                        try:
                            deck_name = f"Vocab_{get_beijing_time_str()}"
                            
                            def update_pkg_progress(ratio: float, text: str) -> None:
                                progress_bar.progress(ratio)
                                status_text.text(text)
                            
                            file_path = generate_anki_package(
                                parsed_data,
                                deck_name,
                                enable_tts=enable_audio_auto,
                                tts_voice=selected_voice_code,
                                progress_callback=update_pkg_progress
                            )
                            
                            with open(file_path, "rb") as f:
                                st.session_state['anki_pkg_data'] = f.read()
                            st.session_state['anki_pkg_name'] = f"{deck_name}.apkg"
                            
                            status_text.markdown(f"✅ **处理完成！共生成 {len(parsed_data)} 张卡片**")
                            st.balloons()
                        except Exception as e:
                            ErrorHandler.handle(e, "生成出错")
                    else:
                        st.error("解析失败，AI 返回内容为空或格式错误。")
                else:
                    st.error("AI 生成失败，请检查 API Key 或网络连接。")
            
            st.caption("⚠️ AI 生成内容可能存在错误，请人工复核。")
        
        if st.session_state.get('anki_pkg_data'):
            st.download_button(
                label=f"📥 立即下载 {st.session_state['anki_pkg_name']}",
                data=st.session_state['anki_pkg_data'],
                file_name=st.session_state['anki_pkg_name'],
                mime="application/octet-stream",
                type="primary",
                use_container_width=True
            )
        
        with col_copy_hint:
            st.info("👈 点击左侧按钮自动生成。如使用第三方 AI，请复制下方 Prompt。")
        
        with st.expander("📌 手动复制 Prompt (第三方 AI 用)"):
            batch_size_prompt = st.number_input("🔢 分组大小 (Max 500)", 10, 500, 50, step=10)
            current_batch_words = []
            
            if words_only:
                total_w = len(words_only)
                num_batches = (total_w + batch_size_prompt - 1) // batch_size_prompt
                batch_options = [
                    f"第 {i+1} 组 ({i*batch_size_prompt+1} - {min((i+1)*batch_size_prompt, total_w)})"
                    for i in range(num_batches)
                ]
                selected_batch_str = st.selectbox("📂 选择当前分组", batch_options)
                sel_idx = batch_options.index(selected_batch_str)
                current_batch_words = words_only[
                    sel_idx*batch_size_prompt:min((sel_idx+1)*batch_size_prompt, total_w)
                ]
            else:
                st.warning("⚠️ 暂无单词数据，请先提取单词。")
            
            words_str_for_prompt = ", ".join(current_batch_words) if current_batch_words else "[INSERT YOUR WORD LIST HERE]"
            
            strict_prompt_template = f"""# Role
You are an expert English Lexicographer and Anki Card Designer. Your goal is to convert a list of target words into high-quality, import-ready Anki flashcards focusing on **natural collocations** (word chunks).
Make sure to process everything in one go, without missing anything.
# Input Data
{words_str_for_prompt}

# Output Format Guidelines
1. **Output Container**: Strictly inside a single ```text code block.
2. **Layout**: One entry per line.
3. **Separator**: Use `|||` as the delimiter.
4. **Target Structure**:
   `Natural Phrase/Collocation` ||| `Concise Definition of the Phrase` ||| `Short Example Sentence` ||| `Etymology breakdown (Simplified Chinese)`

# Field Constraints (Strict)
1. **Field 1: Phrase (CRITICAL)**
   - DO NOT output the single target word.
   - You MUST generate a high-frequency **collocation** or **short phrase** containing the target word.
   - Example: If input is "rain", output "heavy rain" or "torrential rain".
   
2. **Field 2: Definition (English)**
   - Define the *phrase*, not just the isolated word. Keep it concise (B2-C1 level English).

3. **Field 3: Example**
   - A short, authentic sentence containing the phrase.

4. **Field 4: Roots/Etymology (Simplified Chinese)**
   - Format: `prefix- (meaning) + root (meaning) + -suffix (meaning)`.
   - If no classical roots exist, explain the origin briefly in Chinese.
   - Use Simplified Chinese for meanings.

# Valid Example (Follow this logic strictly)
Input: altruism
Output:
motivated by altruism ||| acting out of selfless concern for the well-being of others ||| His donation was motivated by altruism, not a desire for fame. ||| alter (其他) + -ism (主义/行为)

Input: hectic
Output:
a hectic schedule ||| a timeline full of frantic activity and very busy ||| She has a hectic schedule with meetings all day. ||| hect- (持续的/习惯性的 - 来自希腊语hektikos) + -ic (形容词后缀)

# Task
Process the provided input list strictly adhering to the format above."""
            st.code(strict_prompt_template, language="text")

# ==========================================
# Tab 2: Manual Anki Card Creation
# ==========================================
with tab_anki:
    st.markdown("### 📦 手动制作 Anki 牌组")
    
    if 'anki_cards_cache' not in st.session_state:
        st.session_state['anki_cards_cache'] = None
    
    def reset_anki_state() -> None:
        st.session_state['anki_cards_cache'] = None
        st.session_state['anki_pkg_data'] = None
        st.session_state['anki_pkg_name'] = ""
        st.session_state['anki_input_text'] = ""
    
    col_input, col_act = st.columns([3, 1])
    with col_input:
        beijing_time_str = get_beijing_time_str()
        deck_name = st.text_input("🏷️ 牌组名称", f"Vocab_{beijing_time_str}")
    
    ai_response = st.text_area(
        "粘贴 AI 返回内容",
        height=300,
        key="anki_input_text",
        placeholder='hectic ||| 忙乱的 ||| She has a hectic schedule today.'
    )
    
    manual_voice_label = st.radio(
        "🎙️ 发音人",
        options=list(VOICE_MAP.keys()),
        index=0,
        horizontal=True,
        key="sel_voice_manual"
    )
    manual_voice_code = VOICE_MAP[manual_voice_label]
    
    enable_audio = st.checkbox("启用语音", value=True, key="chk_audio_manual")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        start_gen = st.button("🚀 生成卡片", type="primary", use_container_width=True)
    with col_btn2:
        st.button("🗑️ 清空重置", type="secondary", on_click=reset_anki_state, key="btn_clear_anki")
    
    if start_gen:
        if not ai_response.strip():
            st.warning("⚠️ 输入框为空。")
        else:
            progress_container = st.container()
            with progress_container:
                progress_bar_manual = st.progress(0)
                status_manual = st.empty()
            
            def update_progress_manual(ratio: float, text: str) -> None:
                progress_bar_manual.progress(ratio)
                status_manual.text(text)
            
            with st.spinner("⏳ 正在解析并生成..."):
                parsed_data = parse_anki_data(ai_response)
                if parsed_data:
                    st.session_state['anki_cards_cache'] = parsed_data
                    try:
                        file_path = generate_anki_package(
                            parsed_data,
                            deck_name,
                            enable_tts=enable_audio,
                            tts_voice=manual_voice_code,
                            progress_callback=update_progress_manual
                        )
                        
                        with open(file_path, "rb") as f:
                            st.session_state['anki_pkg_data'] = f.read()
                        st.session_state['anki_pkg_name'] = f"{deck_name}.apkg"
                        
                        status_manual.markdown(f"✅ **生成完毕！共制作 {len(parsed_data)} 张卡片**")
                        st.balloons()
                        st.toast("任务完成！", icon="🎉")
                    except Exception as e:
                        ErrorHandler.handle(e, "生成文件出错")
                else:
                    st.error("❌ 解析失败，请检查输入格式。")
    
    if st.session_state['anki_cards_cache']:
        cards = st.session_state['anki_cards_cache']
        with st.expander(f"👀 预览卡片 (前 {MAX_PREVIEW_CARDS} 张)", expanded=True):
            df_view = pd.DataFrame(cards)
            cols = ["正面", "中文/英文释义", "例句"]
            if len(df_view.columns) > 3:
                cols.append("词源")
            df_view.columns = cols[:len(df_view.columns)]
            st.dataframe(df_view.head(MAX_PREVIEW_CARDS), use_container_width=True, hide_index=True)
        
        if st.session_state.get('anki_pkg_data'):
            st.download_button(
                label=f"📥 下载 {st.session_state['anki_pkg_name']}",
                data=st.session_state['anki_pkg_data'],
                file_name=st.session_state['anki_pkg_name'],
                mime="application/octet-stream",
                type="primary"
            )

# ==========================================
# Tab 3: Text-to-Speech (TXT -> Anki)
# ==========================================
with tab_optimize:
    st.markdown("### 🗣️ 文本转语音 (TXT -> Anki)")
    st.info("💡 适合大批量处理，将实时显示生成进度。我们支持最多 4 列数据的映射，确保信息不遗漏。")
    
    uploaded_txt = st.file_uploader(
        "上传 .txt / .csv 文件",
        type=['txt', 'csv'],
        key="txt_audio_up"
    )
    
    if uploaded_txt:
        try:
            # Preprocess file, clean Anki headers
            string_data = uploaded_txt.getvalue().decode("utf-8", errors="ignore")
            lines = string_data.splitlines()
            valid_lines = [line for line in lines if not line.strip().startswith("#")]
            
            if not valid_lines:
                st.error("文件内容为空。")
            else:
                clean_data = "\n".join(valid_lines)
                
                # Detect header
                first_line_clean = valid_lines[0].lower()
                has_header = any(
                    keyword in first_line_clean
                    for keyword in ['word', 'term', 'phrase', 'meaning', 'def', 'example']
                )
                header_arg = 0 if has_header else None
                
                df_preview = pd.read_csv(
                    StringIO(clean_data),
                    sep=None,
                    engine='python',
                    dtype=str,
                    header=header_arg
                ).fillna('')
                
                # Auto-name columns if no header
                if header_arg is None:
                    df_preview.columns = [
                        f"第 {i+1} 列 (示例: {df_preview.iloc[0, i]})"
                        for i in range(len(df_preview.columns))
                    ]
                
                st.toast(f"成功读取 {len(df_preview)} 行数据", icon="✅")
                
                # Column mapping configuration
                st.write("#### 1. 核心步骤：请核对列名")
                st.caption("提示：Prompt 生成了 4 列数据，请务必将'词源'也选上，防止丢失。")
                st.dataframe(df_preview.head(3), use_container_width=True, hide_index=True)
                
                all_cols = list(df_preview.columns)
                all_cols_options = ["(无)"] + all_cols
                
                # 2x2 layout for 4 selection boxes
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                # Smart default indices
                idx_word = 0
                idx_meaning = 1 if len(all_cols) > 1 else 0
                idx_example = 2 if len(all_cols) > 2 else 0
                idx_etym = 3 if len(all_cols) > 3 else 0
                
                col_word = col1.selectbox(
                    "📝 单词/短语列 (正面+语音)",
                    all_cols,
                    index=idx_word
                )
                col_meaning = col2.selectbox(
                    "🇨🇳 释义列 (背面-不发音)",
                    all_cols_options,
                    index=idx_meaning + 1
                )
                col_example = col3.selectbox(
                    "🗣️ 例句列 (背面+语音)",
                    all_cols_options,
                    index=idx_example + 1
                )
                col_etym = col4.selectbox(
                    "🌱 词源/备注列 (背面-不发音)",
                    all_cols_options,
                    index=idx_etym + 1
                )
                
                # Voice configuration
                st.write("#### 2. 生成配置")
                voice_choice_txt = st.radio(
                    "选择发音人",
                    list(VOICE_MAP.keys()),
                    horizontal=True,
                    key="txt_voice_radio"
                )
                voice_code_txt = VOICE_MAP[voice_choice_txt]
                
                txt_deck_name = st.text_input(
                    "牌组名称",
                    f"AudioDeck_{get_beijing_time_str()}",
                    key="txt_deck_name"
                )
                
                # Execute button
                if st.button("🚀 开始生成 (可视化进度)", type="primary", key="btn_txt_gen"):
                    if not col_word:
                        st.error("❌ 必须选择'单词列'！")
                    else:
                        # Prepare data
                        full_cards_list = []
                        for idx, row in df_preview.iterrows():
                            word_val = safe_str_clean(row[col_word])
                            meaning_val = safe_str_clean(row[col_meaning]) if col_meaning != "(无)" else ""
                            example_val = safe_str_clean(row[col_example]) if col_example != "(无)" else ""
                            etym_val = safe_str_clean(row[col_etym]) if col_etym != "(无)" else ""
                            
                            if word_val:
                                full_cards_list.append({
                                    'w': word_val,
                                    'm': meaning_val,
                                    'e': example_val,
                                    'r': etym_val
                                })
                        
                        total_cards = len(full_cards_list)
                        if total_cards == 0:
                            st.warning("有效数据为空。")
                        else:
                            st.divider()
                            st.write(f"📊 任务总量: **{total_cards}** 张卡片")
                            
                            progress_container = st.container()
                            with progress_container:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                            
                            def visual_progress_callback(progress_ratio: float, status_message: str) -> None:
                                progress_bar.progress(progress_ratio)
                                status_text.markdown(f"### {status_message}")
                            
                            try:
                                with st.spinner("正在初始化音频引擎..."):
                                    file_path = generate_anki_package(
                                        full_cards_list,
                                        txt_deck_name,
                                        enable_tts=True,
                                        tts_voice=voice_code_txt,
                                        progress_callback=visual_progress_callback
                                    )
                                
                                with open(file_path, "rb") as f:
                                    st.session_state['txt_pkg_data'] = f.read()
                                st.session_state['txt_pkg_name'] = f"{txt_deck_name}.apkg"
                                
                                status_text.markdown(f"## ✅ 生成完成！共 {total_cards} 张。")
                                progress_bar.progress(1.0)
                                st.balloons()
                                
                            except Exception as e:
                                ErrorHandler.handle(e, "处理失败")
        
        except Exception as e:
            ErrorHandler.handle(e, "系统错误")
    
    if st.session_state.get('txt_pkg_data'):
        st.download_button(
            label=f"📥 下载牌组 {st.session_state['txt_pkg_name']}",
            data=st.session_state['txt_pkg_data'],
            file_name=st.session_state['txt_pkg_name'],
            mime="application/octet-stream",
            type="primary",
            use_container_width=True
        )

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
from collections import Counter
from datetime import datetime, timedelta, timezone

# 尝试导入 OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# 初始化 Session State
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"
if 'anki_input_text' not in st.session_state:
    st.session_state['anki_input_text'] = ""
if 'anki_pkg_data' not in st.session_state: 
    st.session_state['anki_pkg_data'] = None
if 'anki_pkg_name' not in st.session_state: 
    st.session_state['anki_pkg_name'] = ""
if 'opt_pkg_data' not in st.session_state:
    st.session_state['opt_pkg_data'] = None
if 'opt_pkg_name' not in st.session_state:
    st.session_state['opt_pkg_name'] = ""
if 'url_input_key' not in st.session_state:
    st.session_state['url_input_key'] = ""

# 发音人映射
VOICE_MAP = {
    "👩 美音女声 (Jenny)": "en-US-JennyNeural",
    "👨 美音男声 (Christopher)": "en-US-ChristopherNeural",
    "👩 英音女声 (Sonia)": "en-GB-SoniaNeural",
    "👨 英音男声 (Ryan)": "en-GB-RyanNeural",
    "🇨🇳 中文女声 (Xiaoxiao)": "zh-CN-XiaoxiaoNeural"
}

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
# 1. 资源懒加载
# ==========================================
@st.cache_resource(show_spinner="正在加载 NLP 引擎...")
def load_nlp_resources():
    import nltk
    import lemminflect
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
            try: nltk.data.find(f'tokenizers/{pkg}')
            except LookupError: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        st.error(f"NLP 资源加载失败: {e}")
    return nltk, lemminflect

def get_file_parsers():
    import pypdf
    import docx
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    return pypdf, docx, ebooklib, epub, BeautifulSoup

def get_genanki():
    import genanki
    import tempfile
    return genanki, tempfile

@st.cache_data
def load_vocab_data():
    if os.path.exists("vocab.pkl"):
        try:
            df = pd.read_pickle("vocab.pkl")
            return pd.Series(df['rank'].values, index=df['word']).to_dict(), df
        except: pass

    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            w_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            r_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            return pd.Series(df[r_col].values, index=df[w_col]).to_dict(), df
        except: return {}, None
    return {}, None

VOCAB_DICT, FULL_DF = load_vocab_data()

def get_beijing_time_str():
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

def clear_all_state():
    if 'url_input_key' in st.session_state:
        st.session_state['url_input_key'] = ""
    
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'stats_info', 'anki_pkg_data', 'anki_pkg_name', 'anki_input_text', 'opt_pkg_data', 'opt_pkg_name']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))

# ==========================================
# 2. 文本提取逻辑
# ==========================================
def extract_text_from_url(url):
    _, _, _, _, BeautifulSoup = get_file_parsers()
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
            script.decompose()
            
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"Error fetching URL: {e}"

def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    _, tempfile = get_genanki() 
    
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            bytes_data = uploaded_file.getvalue()
            for encoding in ['utf-8', 'gb18030', 'latin-1']:
                try: text = bytes_data.decode(encoding); break
                except: continue
        
        elif file_type == 'pdf':
            try:
                reader = pypdf.PdfReader(uploaded_file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except Exception as e:
                return f"Error parsing PDF: {e}"
        
        elif file_type == 'docx':
            try:
                doc = docx.Document(uploaded_file)
                text = "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                return f"Error parsing DOCX: {e}"
        
        elif file_type == 'epub':
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                book = epub.read_epub(tmp_path)
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        text += soup.get_text(separator=' ', strip=True) + " "
                os.remove(tmp_path)
            except Exception as e:
                 if os.path.exists(tmp_path): os.remove(tmp_path)
                 return f"Error parsing EPUB: {e}"
            
        elif file_type in ['db', 'sqlite']:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_db:
                tmp_db.write(uploaded_file.getvalue())
                tmp_db_path = tmp_db.name
            try:
                conn = sqlite3.connect(tmp_db_path)
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT stem FROM WORDS WHERE stem IS NOT NULL")
                    rows = cursor.fetchall()
                    text = " ".join([r[0] for r in rows if r[0]])
                    if not text:
                         cursor.execute("SELECT word FROM WORDS")
                         rows = cursor.fetchall()
                         text = " ".join([r[0] for r in rows if r[0]])
                except Exception as db_err:
                    text = f"Error reading DB schema: {db_err}"
                conn.close()
            except Exception as e:
                text = f"Error connecting to DB: {e}"
            finally:
                if os.path.exists(tmp_db_path): os.remove(tmp_db_path)

    except Exception as e:
        return f"Unexpected Error: {e}"
    
    return text

def is_valid_word(word):
    if len(word) < 2: return False
    if len(word) > 25: return False 
    if re.search(r'(.)\1{2,}', word): return False
    if not re.search(r'[aeiouy]', word): return False
    return True

def analyze_logic(text, current_lvl, target_lvl, include_unknown):
    nltk, lemminflect = load_nlp_resources()
    
    def get_lemma_local(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_raw_count = len(raw_tokens)
    
    valid_tokens = [t.lower() for t in raw_tokens if is_valid_word(t.lower())]
    token_counts = Counter(valid_tokens)
    
    stats_known_count = 0  
    stats_target_count = 0 
    stats_valid_total = sum(token_counts.values()) 
    
    final_candidates = [] 
    seen_lemmas = set()
    
    for w, count in token_counts.items():
        lemma = get_lemma_local(w)
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        
        if rank_lemma != 99999 and rank_orig != 99999:
            best_rank = min(rank_lemma, rank_orig)
        elif rank_lemma != 99999:
            best_rank = rank_lemma
        else:
            best_rank = rank_orig
            
        if best_rank < current_lvl:
            stats_known_count += count
        elif current_lvl <= best_rank <= target_lvl:
            stats_target_count += count
            
        is_in_range = (best_rank >= current_lvl and best_rank <= target_lvl)
        is_unknown_included = (best_rank == 99999 and include_unknown)
        
        if is_in_range or is_unknown_included:
            word_to_keep = lemma if rank_lemma != 99999 else w
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)
    
    final_candidates.sort(key=lambda x: x[1])
    
    coverage_ratio = (stats_known_count / stats_valid_total) if stats_valid_total > 0 else 0
    target_ratio = (stats_target_count / stats_valid_total) if stats_valid_total > 0 else 0
    
    stats_info = {
        "coverage": coverage_ratio,
        "target_density": target_ratio
    }
    return final_candidates, total_raw_count, stats_info

# ==========================================
# 3. AI 调用逻辑
# ==========================================
def process_ai_in_batches(words_list, progress_callback=None):
    if not OpenAI:
        st.error("❌ 未安装 OpenAI 库，无法使用内置 AI 功能。")
        return None
        
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ 未找到 OPENAI_API_KEY。请在 .streamlit/secrets.toml 中配置。")
        return None
        
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = st.secrets.get("OPENAI_MODEL", "deepseek-chat") 
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    BATCH_SIZE = 10 
    total_words = len(words_list)
    full_results = []
    
    system_prompt = "You are a helpful assistant for vocabulary learning."
    
    for i in range(0, total_words, BATCH_SIZE):
        batch = words_list[i : i + BATCH_SIZE]
        current_batch_str = "\n".join(batch)
        
        user_prompt = f"""
Task: Convert English words to Anki cards.
Format: Word ||| Chinese Meaning ||| English Example
Rules: 
1. Front must be the Word only (Original).
2. Definition must be in Simplified Chinese (Concise & Accurate).
3. Sentence must be in English (Simple & Authentic).
4. Do NOT add etymology or extra fields.

Example:
Input: hectic
Output: hectic ||| 忙乱的，繁忙的 ||| She has a hectic schedule today.

Input:
{current_batch_str}
"""
        max_retries = 3
        for attempt in range(max_retries):
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
                    processed_count = min(i + BATCH_SIZE, total_words)
                    progress_callback(processed_count, total_words)
                
                break 
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)
                    continue
                else:
                    st.error(f"Batch {i//BATCH_SIZE + 1} failed after {max_retries} attempts: {e}")
            
    return "\n".join(full_results)

# ==========================================
# 4. 数据解析与 Anki 打包
# ==========================================
def parse_anki_data(raw_text):
    parsed_cards = []
    text = raw_text.strip()
    
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
        
        w = parts[0].strip()
        m = parts[1].strip()
        e = parts[2].strip() if len(parts) > 2 else ""
        r = parts[3].strip() if len(parts) > 3 else "" 

        if w.lower() in seen_phrases: 
            continue
        seen_phrases.add(w.lower())
        
        parsed_cards.append({
            'w': w, 'm': m, 'e': e, 'r': r
        })

    return parsed_cards

async def _generate_audio_batch(tasks, concurrency=1, progress_callback=None):
    semaphore = asyncio.Semaphore(concurrency)
    total_files = len(tasks)
    completed_files = 0

    async def worker(task):
        nonlocal completed_files
        async with semaphore:
            await asyncio.sleep(random.uniform(1.0, 2.5))
            success = False
            error_msg = ""
            for attempt in range(3):
                try:
                    if not os.path.exists(task['path']):
                        comm = edge_tts.Communicate(task['text'], task['voice'])
                        await comm.save(task['path'])
                        if os.path.exists(task['path']) and os.path.getsize(task['path']) > 100:
                             success = True
                             break
                        else:
                             if os.path.exists(task['path']): os.remove(task['path'])
                             raise Exception("File size too small")
                    else:
                        success = True
                        break
                except Exception as e:
                    error_msg = str(e)
                    await asyncio.sleep(2 * (attempt + 1)) 
            completed_files += 1
            if progress_callback:
                progress_callback(completed_files, total_files)

    jobs = [worker(t) for t in tasks]
    await asyncio.gather(*jobs, return_exceptions=True)

def run_async_batch(tasks, concurrency=1, progress_callback=None):
    if not tasks: return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_generate_audio_batch(tasks, concurrency, progress_callback))
    finally:
        loop.close()

def generate_anki_package(cards_data, deck_name, enable_tts=False, tts_voice="en-US-JennyNeural", progress_callback=None):
    genanki, tempfile = get_genanki()
    media_files = [] 
    
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .meaning { font-size: 20px; font-weight: bold; color: #222; margin-bottom: 15px; text-align: left; }
    .nightMode .meaning { color: #e0e0e0; }
    .example { background: #f7f9fa; padding: 15px; border-left: 5px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 24px; line-height: 1.5; text-align: left; margin-bottom: 15px; }
    .nightMode .example { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    """
    
    MODEL_ID = 1842957301 
    DECK_ID = zlib.adler32(deck_name.encode('utf-8'))

    model = genanki.Model(
        MODEL_ID, 
        'VocabFlow Unified Model',
        fields=[
            {'name': 'Phrase'}, {'name': 'Meaning'},
            {'name': 'Example'}, {'name': 'Etymology'},
            {'name': 'Audio_Phrase'}, {'name': 'Audio_Example'}
        ],
        templates=[{
            'name': 'Vocab Card',
            'qfmt': '<div class="phrase">{{Phrase}}</div><div>{{Audio_Phrase}}</div>', 
            'afmt': '{{FrontSide}}<hr><div class="meaning">{{Meaning}}</div><div class="example">🗣️ {{Example}}</div><div>{{Audio_Example}}</div>',
        }], css=CSS
    )
    
    deck = genanki.Deck(DECK_ID, deck_name)
    tmp_dir = tempfile.gettempdir()
    notes_buffer = []
    audio_tasks = []
    total_words_count = len(cards_data)
    
    for idx, c in enumerate(cards_data):
        phrase = str(c.get('w', ''))
        meaning = str(c.get('m', ''))
        example = str(c.get('e', ''))
        etym = str(c.get('r', ''))
        
        audio_phrase_field = ""
        audio_example_field = ""

        if enable_tts and phrase:
            safe_phrase = re.sub(r'[^a-zA-Z0-9]', '_', phrase)[:20]
            unique_id = int(time.time() * 1000) + random.randint(0, 9999)
            
            f_phrase_name = f"tts_{safe_phrase}_{unique_id}_p.mp3"
            path_phrase = os.path.join(tmp_dir, f_phrase_name)
            audio_tasks.append({'text': phrase, 'path': path_phrase, 'voice': tts_voice})
            media_files.append(path_phrase)
            audio_phrase_field = f"[sound:{f_phrase_name}]"
            
            if example and len(example) > 3:
                f_example_name = f"tts_{safe_phrase}_{unique_id}_e.mp3"
                path_example = os.path.join(tmp_dir, f_example_name)
                audio_tasks.append({'text': example, 'path': path_example, 'voice': tts_voice})
                media_files.append(path_example)
                audio_example_field = f"[sound:{f_example_name}]"
        
        note = genanki.Note(
            model=model, 
            fields=[phrase, meaning, example, etym, audio_phrase_field, audio_example_field]
        )
        notes_buffer.append(note)

    if audio_tasks:
        def internal_progress(curr, total):
            if progress_callback:
                progress_callback(0.1 + (curr/total)*0.8, f"🔊 正在生成语音 ({curr}/{total})...")
        run_async_batch(audio_tasks, concurrency=1, progress_callback=internal_progress)

    for note in notes_buffer:
        deck.add_note(note)

    package = genanki.Package(deck)
    package.media_files = media_files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        package.write_to_file(tmp.name)
    
    for f in media_files:
        try: os.remove(f)
        except: pass
    return tmp.name

# ==========================================
# 5. APKG 优化与语音注入逻辑 (核心修复)
# ==========================================

async def _generate_audio_async(text, voice, output_path):
    """异步生成音频的核心函数"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def generate_audio_safe(text, voice, output_path):
    """
    在 Streamlit/同步环境中安全调用异步 TTS。
    创建一个新的事件循环来执行任务，避免与 Streamlit 的循环冲突。
    """
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.run_until_complete(_generate_audio_async(text, voice, output_path))
        return True
    except Exception as e:
        print(f"Generate Audio Safe Error: {e}")
        return False

def process_apkg_with_audio(uploaded_file, audio_configs, progress_callback=None):
    """
    修复版：为 .apkg 文件添加语音 (解决 asyncio 冲突和数据库锁)
    """
    temp_dir = tempfile.mkdtemp()
    extract_dir = os.path.join(temp_dir, "extracted")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    generated_count = 0
    final_path = None

    try:
        # 1. 解压 APKG
        with zipfile.ZipFile(uploaded_file, 'r') as zf:
            zf.extractall(extract_dir)

        # 2. 读取 media 文件
        media_file_path = os.path.join(extract_dir, "media")
        media_map = {}
        if os.path.exists(media_file_path):
            try:
                with open(media_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        # 兼容非标准JSON (Anki media 文件 key 是 string 数字)
                        media_map = json.loads(content)
            except Exception as e:
                # 尝试其他编码或容错
                try:
                     with open(media_file_path, "r", encoding="cp437") as f:
                        media_map = json.load(f)
                except: pass

        current_indices = []
        for k in media_map.keys():
            try: current_indices.append(int(k))
            except: pass
        
        next_media_id = max(current_indices) + 1 if current_indices else 0

        # 3. 连接数据库
        db_path = os.path.join(extract_dir, "collection.anki2")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, mid, flds FROM notes")
        notes = cursor.fetchall()
        
        cursor.execute("SELECT models FROM col")
        col_res = cursor.fetchone()
        if not col_res: raise Exception("Invalid Anki collection")
        models_json = json.loads(col_res[0])

        total_notes = len(notes)
        
        # 4. 遍历处理
        for idx, (note_id, mid, flds) in enumerate(notes):
            if progress_callback and idx % 5 == 0:
                prog_percent = int((idx / total_notes) * 100)
                progress_callback(prog_percent / 100.0, f"处理进度: {idx+1}/{total_notes}...")

            str_mid = str(mid)
            if str_mid not in models_json: continue
            
            model = models_json[str_mid]
            field_names = [f['name'] for f in model['flds']]
            fields_list = flds.split("\x1f")
            
            note_modified = False

            for config in audio_configs:
                # 兼容旧版 config key
                target_field_name = config.get('target_field') 
                # 或者如果你 UI 传的是 idx，需要在这里转换，这里假设传的是字段名
                # 如果 UI 传的是 src_idx/tgt_idx，请使用下面的逻辑：
                src_idx = config.get('src_idx')
                tgt_idx = config.get('tgt_idx')
                voice = config.get('voice')

                # 如果 config 里是索引 (UI 传递的是索引)
                if src_idx is not None and tgt_idx is not None:
                     if src_idx < len(fields_list) and tgt_idx < len(fields_list):
                        text = fields_list[src_idx]
                        clean_text = re.sub(r'<[^>]+>', '', text).strip()
                        
                        if clean_text and "[sound:" not in fields_list[tgt_idx]:
                            # 生成音频
                            audio_filename = f"tts_{note_id}_{tgt_idx}_{int(time.time())}.mp3"
                            audio_save_path = os.path.join(extract_dir, audio_filename)
                            
                            if generate_audio_safe(clean_text, voice, audio_save_path):
                                media_map[str(next_media_id)] = audio_filename
                                next_media_id += 1
                                fields_list[tgt_idx] += f" [sound:{audio_filename}]"
                                note_modified = True
                                generated_count += 1
                                time.sleep(0.1)

            if note_modified:
                new_flds = "\x1f".join(fields_list)
                cursor.execute("UPDATE notes SET flds=?, mod=? WHERE id=?", 
                               (new_flds, int(time.time()), note_id))

        conn.commit()
        conn.close() # 必须关闭

        with open(media_file_path, "w", encoding="utf-8") as f:
            json.dump(media_map, f)

        # 5. 打包
        output_filename = f"optimized_{int(time.time())}.apkg"
        new_apkg_path = os.path.join(temp_dir, output_filename)
        
        with zipfile.ZipFile(new_apkg_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, extract_dir)
                    zf.write(file_path, arcname)
        
        # 移出临时目录
        final_dest = os.path.join(tempfile.gettempdir(), output_filename)
        shutil.move(new_apkg_path, final_dest)
        final_path = final_dest

    except Exception as e:
        st.error(f"Error: {e}")
        return None, str(e)
    finally:
        try: shutil.rmtree(temp_dir)
        except: pass

    return final_path, generated_count

# ==========================================
# 6. UI 布局
# ==========================================
st.title("⚡️ Vocab Flow Ultra")
st.info(f"Session ID: {st.session_state['uploader_id']} | Time: {get_beijing_time_str()}")

tab1, tab2 = st.tabs(["📝 生词本生成", "🛠️ APKG 语音修复"])

with tab1:
    st.header("1. 导入数据")
    input_method = st.radio("选择输入方式:", ["直接粘贴文本", "上传文件 (PDF/DOCX/EPUB/DB)", "输入 URL 文章"], horizontal=True)
    
    raw_text = ""
    
    if input_method == "直接粘贴文本":
        raw_text = st.text_area("在此粘贴英文文本...", height=200)
    elif input_method == "上传文件 (PDF/DOCX/EPUB/DB)":
        up_file = st.file_uploader("拖入文件", type=['txt','pdf','docx','epub','db','sqlite'])
        if up_file: raw_text = extract_text_from_file(up_file)
    else:
        url_in = st.text_input("输入文章链接:", key="url_input_key")
        if url_in: raw_text = extract_text_from_url(url_in)

    if st.button("开始分析", type="primary"):
        if len(raw_text) < 10:
            st.warning("文本太短！")
        else:
            candidates, total, stats = analyze_logic(raw_text, 1, 15000, False)
            st.session_state['gen_words_data'] = [x[0] for x in candidates]
            st.success(f"发现 {len(candidates)} 个生词")

    if 'gen_words_data' in st.session_state:
        words = st.session_state['gen_words_data']
        st.write(f"预览前 50 个: {', '.join(words[:50])}...")
        
        deck_name = st.text_input("牌组名称", value="My Vocab Deck")
        enable_tts = st.checkbox("生成语音 (微软 TTS)", value=True)
        
        if st.button("🚀 生成 Anki 包"):
            prog_bar = st.progress(0)
            status_txt = st.empty()
            
            # AI 定义
            ai_res = process_ai_in_batches(words[:50], lambda p,t: prog_bar.progress(p/t * 0.5))
            cards = parse_anki_data(ai_res)
            
            # 打包
            pkg_path = generate_anki_package(cards, deck_name, enable_tts, "en-US-JennyNeural", 
                                             lambda p,t: status_txt.text(t))
            
            with open(pkg_path, "rb") as f:
                st.download_button("下载 .apkg", f, file_name=f"{deck_name}.apkg")

with tab2:
    st.header("🛠️ 为现有牌组添加语音")
    up_apkg = st.file_uploader("上传 .apkg 文件", type=['apkg'], key="opt_up")
    
    if up_apkg:
        # 这里需要模拟简单的模型解析来让用户选择字段
        # 为了简化，我们直接让用户输入字段索引 (0-based)
        st.info("请指定卡片中存储'单词'的列索引，和要插入'音频'的目标列索引。 (第一列是 0)")
        col1, col2, col3 = st.columns(3)
        with col1:
            src_idx = st.number_input("单词列索引 (Source)", value=0, min_value=0)
        with col2:
            tgt_idx = st.number_input("音频目标列索引 (Target)", value=1, min_value=0)
        with col3:
            voice_sel = st.selectbox("选择语音", list(VOICE_MAP.keys()))
        
        if st.button("开始修复"):
            audio_configs = [{
                'src_idx': src_idx,
                'tgt_idx': tgt_idx,
                'voice': VOICE_MAP[voice_sel]
            }]
            
            prog_bar = st.progress(0)
            status_txt = st.empty()
            
            with st.spinner("正在处理..."):
                new_path, count = process_apkg_with_audio(
                    up_apkg, 
                    audio_configs, 
                    lambda p, t: (prog_bar.progress(p), status_txt.text(t))
                )
            
            if new_path:
                st.success(f"成功生成 {count} 条语音！")
                with open(new_path, "rb") as f:
                    st.download_button("📥 下载修复后的牌组", f, file_name=f"Fixed_{up_apkg.name}")
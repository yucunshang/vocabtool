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
from io import StringIO
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
if 'txt_pkg_data' not in st.session_state:
    st.session_state['txt_pkg_data'] = None
if 'txt_pkg_name' not in st.session_state:
    st.session_state['txt_pkg_name'] = ""
if 'url_input_key' not in st.session_state:
    st.session_state['url_input_key'] = ""

# 发音人映射 (精简为2种)
VOICE_MAP = {
    "👩 美音女声 (Jenny)": "en-US-JennyNeural",
    "👨 美音男声 (Christopher)": "en-US-ChristopherNeural"
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
    
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'stats_info', 'anki_pkg_data', 'anki_pkg_name', 'anki_input_text', 'txt_pkg_data', 'txt_pkg_name']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

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
    
    # 这里的 System Prompt 可以简单些，主要靠 User Prompt 控制格式
    system_prompt = "You are a helpful assistant for vocabulary learning."
    
    for i in range(0, total_words, BATCH_SIZE):
        batch = words_list[i : i + BATCH_SIZE]
        current_batch_str = "\n".join(batch)
        
        # 使用用户更新后的严格 Prompt 逻辑 (稍作简化以适应 API 内部调用，保持核心指令一致)
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
# 4. 数据解析与 Anki 打包 (新卡片)
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
        
        # 按照 Prompt 格式：Phrase ||| Definition ||| Example ||| Etymology
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

async def _generate_audio_batch(tasks, concurrency=3, progress_callback=None):
    """
    智能优化的音频生成逻辑。
    1. 并发设置为 3 (安全甜点值)，既快又不封号。
    2. 使用短随机抖动 (0.1-0.8s) 模拟真人操作。
    """
    semaphore = asyncio.Semaphore(concurrency)
    total_files = len(tasks)
    completed_files = 0

    async def worker(task):
        nonlocal completed_files
        async with semaphore:
            # 智能抖动
            await asyncio.sleep(random.uniform(0.1, 0.8))
            
            success = False
            error_msg = ""
            
            for attempt in range(3):
                try:
                    if not os.path.exists(task['path']):
                        # 关键：每次重试都重新创建对象
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
                    await asyncio.sleep(1.5 * (attempt + 1)) 
            
            if not success:
                print(f"TTS Failed finally for: {task['text']} | Error: {error_msg}")
            
            completed_files += 1
            if progress_callback:
                progress_callback(completed_files / total_files, f"正在生成 ({completed_files}/{total_files})")

    jobs = [worker(t) for t in tasks]
    await asyncio.gather(*jobs, return_exceptions=True)

def run_async_batch(tasks, concurrency=3, progress_callback=None):
    if not tasks:
        return
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # 这里默认 concurrency=3
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
        def internal_progress(ratio, msg):
            if progress_callback:
                progress_callback(ratio, msg)
        
        run_async_batch(audio_tasks, concurrency=3, progress_callback=internal_progress)

    for note in notes_buffer:
        deck.add_note(note)
    
    if progress_callback:
        progress_callback(1.0, "📦 正在打包 .apkg 文件...")

    package = genanki.Package(deck)
    package.media_files = media_files
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        package.write_to_file(tmp.name)
        for f in media_files:
            try: os.remove(f)
            except: pass
        return tmp.name

# ==========================================
# 6. UI 主程序
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

tab_extract, tab_anki, tab_optimize = st.tabs(["1️⃣ 单词提取", "2️⃣ 卡片制作", "3️⃣ 文本转语音(TXT->Anki)"])

# ----------------- Tab 1: 提取与 AI 生成 -----------------
with tab_extract:
    mode_context, mode_direct, mode_rank = st.tabs(["📄 语境分析", "📝 直接输入", "🔢 词频列表"])
    
    with mode_context:
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略前 N 高频词 (Min Rank)", 1, 20000, 6000, step=100)
        targ = c2.number_input("忽略后 N 低频词 (Max Rank)", 2000, 50000, 10000, step=500)
        
        st.markdown("#### 📥 导入内容")
        
        input_url = st.text_input("🔗 输入文章 URL (自动抓取)", placeholder="[https://www.economist.com/](https://www.economist.com/)...", key="url_input_key")
        
        uploaded_file = st.file_uploader(
            "或直接上传文件", 
            type=['txt', 'pdf', 'docx', 'epub', 'db', 'sqlite'],
            key=st.session_state['uploader_id'], 
            label_visibility="collapsed"
        )
        pasted_text = st.text_area("或在此粘贴文本", height=100, key="paste_key", placeholder="支持直接粘贴文章内容...")
        
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
                    final_data, raw_count, stats_info = analyze_logic(raw_text, curr, targ, False)
                    
                    st.session_state['gen_words_data'] = final_data
                    st.session_state['raw_count'] = raw_count
                    st.session_state['stats_info'] = stats_info
                    st.session_state['process_time'] = time.time() - start_time
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容为空或太短", state="error")
    
    with mode_direct:
        raw_input = st.text_area("✍️ 粘贴单词列表 (每行一个 或 逗号分隔)", height=200, placeholder="altruism\nhectic\nserendipity")
        if st.button("🚀 生成列表", key="btn_direct", type="primary"):
            with st.spinner("正在解析列表..."):
                if raw_input.strip():
                    words = [w.strip() for w in re.split(r'[,\n\t]+', raw_input) if w.strip()]
                    unique_words = []
                    seen = set()
                    for w in words:
                        if w.lower() not in seen:
                            seen.add(w.lower())
                            unique_words.append(w)
                    
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
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("起始排名", 1, 20000, 8000, step=100)
             count = c_b.number_input("数量", 10, 5000, 10, step=10)
             if st.button("🚀 生成列表"):
                 with st.spinner("正在提取..."):
                    if FULL_DF is not None:
                        r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                        w_col = next(c for c in FULL_DF.columns if 'word' in c)
                        subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                        st.session_state['gen_words_data'] = list(zip(subset[w_col], subset[r_col]))
                        st.session_state['raw_count'] = 0
                        st.session_state['stats_info'] = None
        else:
             c_min, c_max, c_cnt = st.columns([1,1,1])
             min_r = c_min.number_input("最小排名", 1, 20000, 12000, step=100)
             max_r = c_max.number_input("最大排名", 1, 25000, 15000, step=100)
             r_count = c_cnt.number_input("抽取数量", 10, 5000, 10, step=10)
             if st.button("🎲 随机抽取"):
                 with st.spinner("正在抽取..."):
                    if FULL_DF is not None:
                        r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                        w_col = next(c for c in FULL_DF.columns if 'word' in c)
                        mask = (FULL_DF[r_col] >= min_r) & (FULL_DF[r_col] <= max_r)
                        candidates = FULL_DF[mask]
                        if len(candidates) > 0:
                            subset = candidates.sample(n=min(r_count, len(candidates))).sort_values(r_col)
                            st.session_state['gen_words_data'] = list(zip(subset[w_col], subset[r_col]))
                            st.session_state['raw_count'] = 0
                            st.session_state['stats_info'] = None

    if st.button("🗑️ 清空重置", type="secondary", on_click=clear_all_state, key="btn_clear_extract"): pass

    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]
        
        st.divider()
        st.markdown("### 📊 分析报告")
        k1, k2, k3, k4 = st.columns(4)
        raw_c = st.session_state.get('raw_count', 0)
        stats = st.session_state.get('stats_info', {})
        k1.metric("总词数", f"{raw_c:,}")
        k2.metric("熟词覆盖", f"{stats.get('coverage', 0):.1%}" if stats else "--")
        k3.metric("生词密度", f"{stats.get('target_density', 0):.1%}" if stats else "--")
        k4.metric("提取生词", f"{len(words_only)}")
        
        display_text = ", ".join(words_only)
        with st.expander("📋 预览所有单词", expanded=False):
            st.code(display_text, language="text")

        st.divider()
        st.subheader("🤖 一键生成 Anki 牌组")
        
        st.write("🎙️ **语音设置**")
        
        # 修复冲突: 给 Tab1 的 radio 增加 key
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
                MAX_AUTO_LIMIT = 300 
                target_words = words_only[:MAX_AUTO_LIMIT]
                
                if len(words_only) > MAX_AUTO_LIMIT:
                    st.warning(f"⚠️ 单词过多，自动截取前 {MAX_AUTO_LIMIT} 个进行处理。")
                
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                def update_ai_progress(current, total):
                    # API 调用中的 progress callback 简单适配
                    # current/total 是 batch 级别
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
                            
                            def update_pkg_progress(ratio, text):
                                progress_bar.progress(ratio)
                                status_text.text(text)

                            f_path = generate_anki_package(
                                parsed_data, 
                                deck_name, 
                                enable_tts=enable_audio_auto, 
                                tts_voice=selected_voice_code,
                                progress_callback=update_pkg_progress
                            )
                            with open(f_path, "rb") as f:
                                st.session_state['anki_pkg_data'] = f.read()
                            st.session_state['anki_pkg_name'] = f"{deck_name}.apkg"
                            
                            status_text.markdown(f"✅ **处理完成！共生成 {len(parsed_data)} 张卡片**")
                            st.balloons()
                        except Exception as e:
                            st.error(f"生成出错: {e}")
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
                batch_options = [f"第 {i+1} 组 ({i*batch_size_prompt+1} - {min((i+1)*batch_size_prompt, total_w)})" for i in range(num_batches)]
                selected_batch_str = st.selectbox("📂 选择当前分组", batch_options)
                sel_idx = batch_options.index(selected_batch_str)
                current_batch_words = words_only[sel_idx*batch_size_prompt : min((sel_idx+1)*batch_size_prompt, total_w)]
            else:
                st.warning("⚠️ 暂无单词数据，请先提取单词。")

            words_str_for_prompt = ", ".join(current_batch_words) if current_batch_words else "[INSERT YOUR WORD LIST HERE]"
            
            # === 这里已经更新为你提供的最新 Prompt ===
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

# ----------------- Tab 2: 卡片制作 (手动模式) -----------------
with tab_anki:
    st.markdown("### 📦 手动制作 Anki 牌组")
    
    if 'anki_cards_cache' not in st.session_state: st.session_state['anki_cards_cache'] = None
    
    def reset_anki_state():
        st.session_state['anki_cards_cache'] = None
        st.session_state['anki_pkg_data'] = None
        st.session_state['anki_pkg_name'] = ""
        st.session_state['anki_input_text'] = ""

    col_input, col_act = st.columns([3, 1])
    with col_input:
        bj_time_str = get_beijing_time_str()
        deck_name = st.text_input("🏷️ 牌组名称", f"Vocab_{bj_time_str}")
    
    ai_resp = st.text_area(
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

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        start_gen = st.button("🚀 生成卡片", type="primary", use_container_width=True)
    with c_btn2:
        st.button("🗑️ 清空重置", type="secondary", on_click=reset_anki_state, key="btn_clear_anki")

    if start_gen:
        if not ai_resp.strip():
            st.warning("⚠️ 输入框为空。")
        else:
            prog_cont = st.container()
            with prog_cont:
                progress_bar_manual = st.progress(0)
                status_manual = st.empty()
            
            def update_progress_manual(ratio, text):
                progress_bar_manual.progress(ratio)
                status_manual.text(text)

            with st.spinner("⏳ 正在解析并生成..."):
                parsed_data = parse_anki_data(ai_resp)
                if parsed_data:
                    st.session_state['anki_cards_cache'] = parsed_data
                    try:
                        f_path = generate_anki_package(
                            parsed_data, 
                            deck_name, 
                            enable_tts=enable_audio, 
                            tts_voice=manual_voice_code,
                            progress_callback=update_progress_manual
                        )
                        with open(f_path, "rb") as f:
                            st.session_state['anki_pkg_data'] = f.read()
                        st.session_state['anki_pkg_name'] = f"{deck_name}.apkg"
                        status_manual.markdown(f"✅ **生成完毕！共制作 {len(parsed_data)} 张卡片**")
                        st.balloons()
                        st.toast("任务完成！", icon="🎉")
                    except Exception as e:
                        st.error(f"生成文件出错: {e}")
                else:
                    st.error("❌ 解析失败，请检查输入格式。")

    if st.session_state['anki_cards_cache']:
        cards = st.session_state['anki_cards_cache']
        with st.expander("👀 预览卡片 (前 10 张)", expanded=True):
            df_view = pd.DataFrame(cards)
            cols = ["正面", "中文/英文释义", "例句"]
            if len(df_view.columns) > 3: cols.append("词源")
            df_view.columns = cols[:len(df_view.columns)]
            st.dataframe(df_view.head(10), use_container_width=True, hide_index=True)

        if st.session_state.get('anki_pkg_data'):
            st.download_button(
                label=f"📥 下载 {st.session_state['anki_pkg_name']}",
                data=st.session_state['anki_pkg_data'],
                file_name=st.session_state['anki_pkg_name'],
                mime="application/octet-stream",
                type="primary"
            )

# ----------------- Tab 3: 文本转语音 (TXT -> Anki) -----------------
with tab_optimize:
    st.markdown("### 🗣️ 文本转语音 (TXT -> Anki)")
    st.info("💡 适合大批量处理，将实时显示生成进度。我们支持最多 4 列数据的映射，确保信息不遗漏。")

    up_txt = st.file_uploader("上传 .txt / .csv 文件", type=['txt', 'csv'], key="txt_audio_up")
    
    if up_txt:
        try:
            # === 1. 预处理文件，清洗掉 Anki 的 header ===
            string_data = up_txt.getvalue().decode("utf-8", errors="ignore")
            lines = string_data.splitlines()
            valid_lines = [line for line in lines if not line.strip().startswith("#")]
            
            if not valid_lines:
                st.error("文件内容为空。")
            else:
                clean_data = "\n".join(valid_lines)
                # 简单推断是否有表头
                first_line_clean = valid_lines[0].lower()
                has_header = any(x in first_line_clean for x in ['word', 'term', 'phrase', 'meaning', 'def', 'example'])
                header_arg = 0 if has_header else None
                
                from io import StringIO
                df_preview = pd.read_csv(
                    StringIO(clean_data), 
                    sep=None, 
                    engine='python', 
                    dtype=str, 
                    header=header_arg
                ).fillna('')
                
                # 无表头时自动命名
                if header_arg is None:
                    df_preview.columns = [f"第 {i+1} 列 (示例: {df_preview.iloc[0, i]})" for i in range(len(df_preview.columns))]

                st.toast(f"成功读取 {len(df_preview)} 行数据", icon="✅")
                
                # === 2. 列映射配置 (新增第4列映射) ===
                st.write("#### 1. 核心步骤：请核对列名")
                st.caption("提示：Prompt 生成了 4 列数据，请务必将“词源”也选上，防止丢失。")
                st.dataframe(df_preview.head(3), use_container_width=True, hide_index=True)
                
                all_cols = list(df_preview.columns)
                all_cols_options = ["(无)"] + all_cols
                
                # 使用 2x2 布局以容纳 4 个选择框
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                
                # 智能尝试索引
                idx_word = 0
                idx_meaning = 1 if len(all_cols) > 1 else 0
                idx_example = 2 if len(all_cols) > 2 else 0
                idx_etym = 3 if len(all_cols) > 3 else 0
                
                col_word = c1.selectbox("📝 单词/短语列 (正面+语音)", all_cols, index=idx_word)
                col_meaning = c2.selectbox("🇨🇳 释义列 (背面-不发音)", all_cols_options, index=idx_meaning + 1)
                col_example = c3.selectbox("🗣️ 例句列 (背面+语音)", all_cols_options, index=idx_example + 1)
                col_etym = c4.selectbox("🌱 词源/备注列 (背面-不发音)", all_cols_options, index=idx_etym + 1)
                
                # === 3. 语音配置 ===
                st.write("#### 2. 生成配置")
                voice_choice_txt = st.radio(
                    "选择发音人", 
                    list(VOICE_MAP.keys()), 
                    horizontal=True,
                    key="txt_voice_radio"
                )
                voice_code_txt = VOICE_MAP[voice_choice_txt]
                
                txt_deck_name = st.text_input("牌组名称", f"AudioDeck_{get_beijing_time_str()}", key="txt_deck_name")
                
                # === 4. 执行按钮 ===
                if st.button("🚀 开始生成 (可视化进度)", type="primary", key="btn_txt_gen"):
                    if not col_word:
                        st.error("❌ 必须选择“单词列”！")
                    else:
                        # 准备数据
                        full_cards_list = []
                        for idx, row in df_preview.iterrows():
                            w_val = str(row[col_word]).strip()
                            m_val = str(row[col_meaning]).strip() if col_meaning != "(无)" else ""
                            e_val = str(row[col_example]).strip() if col_example != "(无)" else ""
                            # 新增词源映射
                            r_val = str(row[col_etym]).strip() if col_etym != "(无)" else ""
                            
                            if w_val:
                                full_cards_list.append({
                                    'w': w_val, 
                                    'm': m_val, 
                                    'e': e_val, 
                                    'r': r_val # 传递第4列数据
                                })
                        
                        total_cards = len(full_cards_list)
                        if total_cards == 0:
                            st.warning("有效数据为空。")
                        else:
                            st.divider()
                            st.write(f"📊 任务总量: **{total_cards}** 张卡片")
                            
                            prog_container = st.container()
                            with prog_container:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                            
                            def visual_progress_callback(progress_ratio, status_message):
                                progress_bar.progress(progress_ratio)
                                status_text.markdown(f"### {status_message}")
                                
                            try:
                                with st.spinner("正在初始化音频引擎..."):
                                    f_path = generate_anki_package(
                                        full_cards_list, 
                                        txt_deck_name, 
                                        enable_tts=True, 
                                        tts_voice=voice_code_txt,
                                        progress_callback=visual_progress_callback
                                    )
                                
                                with open(f_path, "rb") as f:
                                    st.session_state['txt_pkg_data'] = f.read()
                                st.session_state['txt_pkg_name'] = f"{txt_deck_name}.apkg"
                                
                                status_text.markdown(f"## ✅ 生成完成！共 {total_cards} 张。")
                                progress_bar.progress(1.0)
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"处理失败: {e}")

        except Exception as e:
            st.error(f"系统错误: {e}")

    if st.session_state.get('txt_pkg_data'):
        st.download_button(
            label=f"📥 下载牌组 {st.session_state['txt_pkg_name']}",
            data=st.session_state['txt_pkg_data'],
            file_name=st.session_state['txt_pkg_name'],
            mime="application/octet-stream",
            type="primary",
            use_container_width=True
        )
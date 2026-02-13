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
from collections import Counter
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# 尝试导入 OpenAI，防止未安装报错
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

# 全局发音人映射
VOICE_MAP = {
    "👩 女声 (Jenny)": "en-US-JennyNeural",
    "👨 男声 (Christopher)": "en-US-ChristopherNeural",
    "🇬🇧 英音女 (Libby)": "en-GB-LibbyNeural",
    "🇬🇧 英音男 (Ryan)": "en-GB-RyanNeural"
}

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; }
    .scrollable-text {
        max-height: 200px; overflow-y: auto; padding: 10px;
        border: 1px solid #eee; border-radius: 5px; background-color: #fafafa;
        font-family: monospace; white-space: pre-wrap;
    }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
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
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'stats_info', 'anki_pkg_data', 'anki_pkg_name', 'anki_input_text', 'anki_cards_cache']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

# ==========================================
# 2. 文本提取逻辑
# ==========================================
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
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type == 'epub':
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
        return f"Error: {e}"
    
    return text

def is_valid_word(word):
    if len(word) < 2 or len(word) > 25: return False 
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
        
        best_rank = min(rank_lemma, rank_orig) if (rank_lemma != 99999 and rank_orig != 99999) else (rank_lemma if rank_lemma != 99999 else rank_orig)
            
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
    
    stats_info = {
        "coverage": (stats_known_count / stats_valid_total) if stats_valid_total > 0 else 0,
        "target_density": (stats_target_count / stats_valid_total) if stats_valid_total > 0 else 0
    }
    return final_candidates, total_raw_count, stats_info

# ==========================================
# 3. AI 调用逻辑 (多线程并发 + Phrase Prompt)
# ==========================================
def process_ai_in_batches(words_list, progress_callback=None):
    """
    多线程并发调用 AI，使用自定义的 Natural Phrase Prompt
    """
    if not OpenAI:
        st.error("❌ 未安装 OpenAI 库，无法使用内置 AI 功能。")
        return None
        
    api_key = st.secrets.get("OPENAI_API_KEY")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = st.secrets.get("OPENAI_MODEL", "gpt-3.5-turbo")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # === 1. 参数配置 (每批5个，并发5个，速度起飞) ===
    BATCH_SIZE = 5
    MAX_WORKERS = 5
    
    batches = [words_list[i:i + BATCH_SIZE] for i in range(0, len(words_list), BATCH_SIZE)]
    total_batches = len(batches)
    full_results = [""] * total_batches 
    
    # === 2. 定义单个任务 ===
    def process_single_batch(index, batch_words):
        batch_str = ", ".join(batch_words)
        # === 植入你的专用 Prompt ===
        user_prompt = f"""
# Role
You are an expert English Lexicographer and Anki Card Designer.
# Input Data
{batch_str}

# Output Format Guidelines
1. **Output Container**: Strictly inside a single ```text code block.
2. **Layout**: One entry per line.
3. **Separator**: Use `|||` as the delimiter.
4. **Target Structure**:
   `Natural Phrase/Collocation` ||| `Concise Definition of the Phrase (English)` ||| `Short Example Sentence` ||| `Etymology breakdown (Simplified Chinese)`

# Field Constraints
1. **Phrase**: High-frequency **collocation** containing the target word.
2. **Definition**: Define the *phrase*. Keep it concise.
3. **Example**: Short, authentic sentence.
4. **Etymology**: `prefix- (meaning) + root (meaning) + -suffix (meaning)`. If simpler, just explain origin in Chinese.

# Example
Input: hectic
Output:
a hectic schedule ||| a timeline full of frantic activity and very busy ||| She has a hectic schedule today. ||| hect- (持续的 - 希腊语hektikos) + -ic (形容词后缀)

# Task
Process strictly. Do not miss any word.
"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful dictionary assistant."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            return index, response.choices[0].message.content
        except Exception as e:
            return index, f"Error: {e}"

    # === 3. 多线程执行 ===
    completed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(process_single_batch, i, batch): i for i, batch in enumerate(batches)}
        
        for future in as_completed(future_to_batch):
            idx, result = future.result()
            full_results[idx] = result
            
            completed_count += 1
            if progress_callback:
                # 估算已处理单词数
                current_words = min(completed_count * BATCH_SIZE, len(words_list))
                progress_callback(current_words, len(words_list))
                
    return "\n".join(full_results)

# ==========================================
# 4. 数据解析与 TTS (异步并发)
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
        if not line or "|||" not in line: continue
            
        parts = line.split("|||")
        if len(parts) < 2: continue
        
        # 映射 Prompt 的 4 个字段
        w = parts[0].strip() # Phrase
        m = parts[1].strip() # Definition
        e = parts[2].strip() if len(parts) > 2 else "" # Example
        r = parts[3].strip() if len(parts) > 3 else "" # Etymology

        if w.lower() in seen_phrases: continue
        seen_phrases.add(w.lower())
        
        parsed_cards.append({'w': w, 'm': m, 'e': e, 'r': r})

    return parsed_cards

async def batch_generate_audio(tasks, progress_callback=None, total_tasks=0):
    """
    异步并发生成音频，限制并发数为 15
    """
    semaphore = asyncio.Semaphore(15) 
    completed = 0
    
    async def limited_task(text, path, voice):
        nonlocal completed
        async with semaphore:
            try:
                if len(text) > 200: text = text[:200] # 防止过长文本报错
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(path)
            except Exception as e:
                print(f"TTS Fail: {e}")
            finally:
                completed += 1
                if progress_callback:
                    # 更新进度条
                    progress_callback(completed / total_tasks, f"🔊 音频生成中... ({completed}/{total_tasks})")

    await asyncio.gather(*(limited_task(*t) for t in tasks))

def generate_anki_package(cards_data, deck_name, enable_tts=False, tts_voice="en-US-JennyNeural", progress_callback=None):
    genanki, tempfile = get_genanki()
    media_files = [] 
    tmp_dir = tempfile.gettempdir()
    
    # 样式配置
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .meaning { font-size: 20px; font-weight: bold; color: #222; margin-bottom: 15px; text-align: left; }
    .nightMode .meaning { color: #e0e0e0; }
    .example { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 18px; text-align: left; margin-bottom: 15px; }
    .nightMode .example { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .etymology { display: block; font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; border: 1px solid #fef3c7; }
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
    """
    
    MODEL_ID = 1842957301
    DECK_ID = zlib.adler32(deck_name.encode('utf-8'))

    model = genanki.Model(
        MODEL_ID, 'VocabFlow Phrase Model',
        fields=[
            {'name': 'Phrase'}, {'name': 'Meaning'},
            {'name': 'Example'}, {'name': 'Etymology'},
            {'name': 'Audio_Phrase'}, {'name': 'Audio_Example'}
        ],
        templates=[{
            'name': 'Phrase Card',
            'qfmt': '<div class="phrase">{{Phrase}}</div><div>{{Audio_Phrase}}</div>', 
            'afmt': '{{FrontSide}}<hr><div class="meaning">{{Meaning}}</div><div class="example">🗣️ {{Example}}</div><div>{{Audio_Example}}</div>{{#Etymology}}<div class="etymology">🌱 词源: {{Etymology}}</div>{{/Etymology}}',
        }], css=CSS
    )
    
    deck = genanki.Deck(DECK_ID, deck_name)
    
    tts_tasks = [] 
    notes_to_add = []

    # 1. 准备所有卡片数据和 TTS 任务
    for idx, c in enumerate(cards_data):
        phrase = str(c.get('w', '')).strip()
        meaning = str(c.get('m', ''))
        example = str(c.get('e', ''))
        etym = str(c.get('r', ''))
        
        audio_phrase_field = ""
        audio_example_field = ""
        
        if enable_tts and phrase:
            safe_phrase = re.sub(r'[^a-zA-Z0-9]', '_', phrase)[:30]
            unique_id = int(time.time() * 1000) + idx
            
            f_phrase_name = f"t_{unique_id}_p.mp3"
            f_example_name = f"t_{unique_id}_e.mp3"
            
            path_phrase = os.path.join(tmp_dir, f_phrase_name)
            path_example = os.path.join(tmp_dir, f_example_name)
            
            tts_tasks.append((phrase, path_phrase, tts_voice))
            media_files.append(path_phrase)
            audio_phrase_field = f"[sound:{f_phrase_name}]"
            
            if example and len(example) > 3:
                tts_tasks.append((example, path_example, tts_voice))
                media_files.append(path_example)
                audio_example_field = f"[sound:{f_example_name}]"

        notes_to_add.append(genanki.Note(
            model=model, 
            fields=[phrase, meaning, example, etym, audio_phrase_field, audio_example_field]
        ))

    # 2. 批量并发生成音频
    if tts_tasks:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(batch_generate_audio(tts_tasks, progress_callback, len(tts_tasks)))
        finally:
            loop.close()
    
    # 3. 添加笔记并打包
    if progress_callback: progress_callback(0.95, "📦 正在打包 .apkg 文件...")
    for note in notes_to_add:
        deck.add_note(note)

    package = genanki.Package(deck)
    package.media_files = [f for f in media_files if os.path.exists(f)]
    
    output_path = os.path.join(tmp_dir, f"{deck_name}.apkg")
    package.write_to_file(output_path)
    return output_path

# ==========================================
# 5. UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra (Turbo)")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv` 文件，请检查目录。")

with st.expander("📖 使用指南"):
    st.markdown("""
    **🚀 极速工作流 (已优化)**
    1. **提取**：上传文件或粘贴文本。
    2. **生成**：点击“使用内置 AI 生成”。**现已支持多线程并发**，速度极大提升。
    3. **下载**：音频生成采用**异步并发**，无需漫长等待。
    """)

tab_extract, tab_anki = st.tabs(["1️⃣ 单词提取", "2️⃣ 卡片制作"])

# ----------------- Tab 1: 提取与 AI 生成 -----------------
with tab_extract:
    mode_context, mode_direct, mode_rank = st.tabs(["📄 语境分析", "📝 直接输入", "🔢 词频列表"])
    
    # 模式1：语境分析
    with mode_context:
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略前 N 高频词", 1, 20000, 6000, step=100)
        targ = c2.number_input("忽略后 N 低频词", 2000, 50000, 10000, step=500)
        
        uploaded_file = st.file_uploader("直接上传文件", key=st.session_state['uploader_id'])
        pasted_text = st.text_area("或在此粘贴文本", height=100, key="paste_key")
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("正在处理中...", expanded=True) as status:
                start_time = time.time()
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                
                if len(raw_text) > 2:
                    status.write("🔍 正在分析...")
                    final_data, raw_count, stats_info = analyze_logic(raw_text, curr, targ, False)
                    
                    st.session_state['gen_words_data'] = final_data
                    st.session_state['raw_count'] = raw_count
                    st.session_state['stats_info'] = stats_info
                    st.session_state['process_time'] = time.time() - start_time
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容为空或太短", state="error")
    
    # 模式2：直接输入
    with mode_direct:
        raw_input = st.text_area("✍️ 粘贴单词列表", height=200, placeholder="altruism\nhectic\nserendipity")
        if st.button("🚀 生成列表", key="btn_direct", type="primary"):
            if raw_input.strip():
                words = [w.strip() for w in re.split(r'[,\n\t]+', raw_input) if w.strip()]
                unique_words = sorted(list(set(words)), key=words.index)
                st.session_state['gen_words_data'] = [(w, VOCAB_DICT.get(w.lower(), 99999)) for w in unique_words]
                st.session_state['raw_count'] = len(unique_words)
                st.session_state['stats_info'] = None 
                st.success(f"✅ 已加载 {len(unique_words)} 个单词")

    # 模式3：词频列表
    with mode_rank:
        gen_type = st.radio("生成模式", ["🔢 顺序生成", "🔀 随机抽取"], horizontal=True)
        if "顺序生成" in gen_type:
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("起始排名", 1, 20000, 8000, step=100)
             count = c_b.number_input("数量", 10, 5000, 50, step=50)
             if st.button("🚀 生成列表"):
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
             r_count = c_cnt.number_input("抽取数量", 10, 5000, 50, step=50)
             if st.button("🎲 随机抽取"):
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

    # --- 结果展示与 AI 生成区 ---
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
        
        with st.expander("📋 预览单词列表", expanded=False):
            st.code(", ".join(words_only), language="text")

        st.divider()
        st.subheader("🤖 一键生成 Anki 牌组 (Turbo)")
        
        st.write("🎙️ **语音设置**")
        c_voice, c_check = st.columns([3, 1])
        with c_voice:
            selected_voice_label = st.radio("选择发音人", options=list(VOICE_MAP.keys()), index=0, horizontal=True, label_visibility="collapsed")
            selected_voice_code = VOICE_MAP[selected_voice_label]
        with c_check:
            enable_audio_auto = st.checkbox("启用语音", value=True, key="chk_audio_auto")

        col_ai_btn, col_copy_hint = st.columns([1, 2])
        
        with col_ai_btn:
            if st.button("✨ 使用内置 AI 生成", type="primary", use_container_width=True):
                MAX_AUTO_LIMIT = 300 
                target_words = words_only[:MAX_AUTO_LIMIT]
                
                if len(words_only) > MAX_AUTO_LIMIT:
                    st.warning(f"⚠️ 单词过多，自动截取前 {MAX_AUTO_LIMIT} 个。")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_ai_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.markdown(f"🤖 **AI 正在并发思考... ({current}/{total})**")

                # 多线程 AI 生成
                ai_result = process_ai_in_batches(target_words, progress_callback=update_ai_progress)
                
                if ai_result:
                    st.session_state['anki_input_text'] = ai_result
                    parsed_data = parse_anki_data(ai_result)
                    
                    if parsed_data:
                        try:
                            deck_name = f"Vocab_{get_beijing_time_str()}"
                            
                            def update_pkg_progress(p, text):
                                progress_bar.progress(p)
                                status_text.text(text)

                            # 异步 TTS 生成
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
                        st.error("解析失败，AI 返回内容为空。")
                else:
                    st.error("AI 连接失败。")

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
            st.info("👈 提示：现在支持多线程加速，速度提升约 5 倍！")

        with st.expander("📌 手动复制 Prompt (第三方 AI 用)"):
            st.code("""# Role
You are an expert English Lexicographer.
# Output Format
Phrase ||| Definition (English) ||| Example ||| Etymology (Chinese)
# Task
Convert list to natural collocations.""", language="text")

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
        deck_name = st.text_input("🏷️ 牌组名称", f"Vocab_{get_beijing_time_str()}")
    
    ai_resp = st.text_area(
        "粘贴 AI 返回内容", 
        height=300, 
        key="anki_input_text",
        placeholder='a hectic schedule ||| very busy ||| She has a hectic schedule. ||| hect- (持续的)'
    )
    
    col_voice_opt, col_voice_sw = st.columns([3, 1])
    with col_voice_opt:
        manual_voice_label = st.radio("🎙️ 发音人", options=list(VOICE_MAP.keys()), index=0, horizontal=True, key="sel_voice_manual")
        manual_voice_code = VOICE_MAP[manual_voice_label]

    with col_voice_sw:
        st.write("")
        st.write("")
        enable_audio = st.checkbox("启用语音", value=True, key="chk_audio_manual")

    if st.button("🚀 生成卡片", type="primary"):
        if ai_resp.strip():
            progress_bar_manual = st.progress(0)
            status_manual = st.empty()
            
            def update_progress_manual(p, text):
                progress_bar_manual.progress(p)
                status_manual.text(text)

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
                    status_manual.text("✅ 生成完毕！")
                except Exception as e:
                    st.error(f"生成文件出错: {e}")
            else:
                st.error("❌ 解析失败。")

    if st.session_state['anki_cards_cache']:
        cards = st.session_state['anki_cards_cache']
        with st.expander("👀 预览卡片 (前 50 张)", expanded=True):
            st.dataframe(pd.DataFrame(cards).rename(columns={'w':'Phrase','m':'Def','e':'Ex','r':'Etym'}), use_container_width=True, hide_index=True)

        if st.session_state.get('anki_pkg_data'):
            st.download_button(
                label=f"📥 下载 {st.session_state['anki_pkg_name']}",
                data=st.session_state['anki_pkg_data'],
                file_name=st.session_state['anki_pkg_name'],
                mime="application/octet-stream",
                type="primary"
            )
import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
import sqlite3
import tempfile
import zlib
from collections import Counter
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# 动态 Key 初始化 (用于一键清空)
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    .stat-box { padding: 15px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center; color: #166534; margin-bottom: 20px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; }
    
    /* 滚动容器样式 */
    .scrollable-text {
        max-height: 200px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 5px;
        background-color: #fafafa;
        font-family: monospace;
        white-space: pre-wrap;
    }
    
    /* 指南样式 */
    .guide-step { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0056b3; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .guide-title { font-size: 18px; font-weight: bold; color: #0f172a; margin-bottom: 10px; display: block; }
    .guide-tip { font-size: 14px; color: #64748b; background: #eef2ff; padding: 8px; border-radius: 4px; margin-top: 8px; }
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
    except: pass
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
    """
    加载 COCA 词频表
    """
    possible_files = ["coca_cleaned.csv", "data.csv", "vocab.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            w_col = next((c for c in df.columns if 'word' in c), df.columns[0])
            r_col = next((c for c in df.columns if 'rank' in c), df.columns[1])
            df = df.dropna(subset=[w_col])
            # 统一转小写，去空格
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
    """
    强力清空：重置所有状态
    """
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'stats_info', 'paste_key']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    
    # 重置粘贴框
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

# ==========================================
# 2. 核心逻辑 (修复版)
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    text = ""
    # 获取文件名后缀 (小写)
    file_name = uploaded_file.name.lower()
    file_type = file_name.split('.')[-1]
    
    try:
        # --- Kindle 生词本 (.db) ---
        if file_type == 'db':
            # 需要保存为临时文件才能被 sqlite3 读取
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                conn = sqlite3.connect(tmp_path)
                cursor = conn.cursor()
                
                # 【关键修复】取消 WHERE lang='en' 限制
                # 很多 Kindle 书籍的语言标记为空或 en-US，限制会导致提取为空
                cursor.execute("SELECT stem FROM WORDS")
                
                rows = cursor.fetchall()
                # 拼接成字符串
                text = " ".join([row[0] for row in rows if row[0]])
                
                if not text.strip():
                    return "ERROR: 数据库读取成功，但未找到单词。请确认这是一个有效的 Kindle 生词本。"
                    
            except Exception as e:
                return f"ERROR_DB: 数据库读取错误 - {e}"
            finally:
                if 'conn' in locals(): conn.close()
                if os.path.exists(tmp_path): os.remove(tmp_path)

        # --- TXT 处理 ---
        elif file_type == 'txt':
            bytes_data = uploaded_file.getvalue()
            decoded_text = ""
            for encoding in ['utf-8', 'utf-8-sig', 'gb18030', 'latin-1']:
                try:
                    decoded_text = bytes_data.decode(encoding)
                    break
                except: continue
            
            # 检测是否为 Kindle My Clippings
            if "==========" in decoded_text:
                clips = []
                entries = decoded_text.split("==========")
                for entry in entries:
                    lines = [l.strip() for l in entry.split('\n') if l.strip()]
                    if len(lines) >= 3:
                        clips.append(lines[-1])
                text = "\n".join(clips)
            else:
                text = decoded_text

        # --- PDF 处理 ---
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
        # --- DOCX 处理 ---
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
            
        # --- EPUB 处理 ---
        elif file_type == 'epub':
            genanki, tempfile = get_genanki()
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
        return f"ERROR_FILE: {e}"
    return text

def is_valid_word(word):
    """
    垃圾词清洗
    """
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
# (优化版) JSON 解析逻辑
# ==========================================
def parse_anki_data(raw_text):
    """
    鲁棒的 JSON 解析器
    """
    parsed_cards = []
    text = raw_text.strip()
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'```', '', text).strip()
    
    json_objects = []

    # 策略 A: 尝试作为完整列表
    try:
        data = json.loads(text)
        if isinstance(data, list):
            json_objects = data
    except:
        # 策略 B: 扫描式解析 (NDJSON)
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            while pos < len(text) and (text[pos].isspace() or text[pos] == ','):
                pos += 1
            if pos >= len(text): break
            try:
                obj, index = decoder.raw_decode(text[pos:])
                json_objects.append(obj)
                pos += index
            except: pos += 1

    seen_phrases_lower = set()

    for data in json_objects:
        if not isinstance(data, dict): continue
            
        def get_val(keys_list):
            for k in keys_list:
                if k in data: return data[k]
                for data_k in data.keys():
                    if data_k.lower() == k.lower(): return data[data_k]
            return ""

        front_text = get_val(['w', 'word', 'phrase', 'term'])
        meaning = get_val(['m', 'meaning', 'def', 'definition'])
        examples = get_val(['e', 'example', 'examples', 'sentence'])
        etymology = get_val(['r', 'root', 'etymology', 'origin'])

        if not front_text or not meaning: continue
        
        front_text = str(front_text).replace('**', '').strip()
        meaning = str(meaning).strip()
        examples = str(examples).strip()
        etymology = str(etymology).strip()
        
        if etymology.lower() in ["none", "null", ""]: etymology = ""

        if front_text.lower() in seen_phrases_lower: continue
        seen_phrases_lower.add(front_text.lower())

        parsed_cards.append({
            'front_phrase': front_text,
            'meaning': meaning,
            'examples': examples,
            'etymology': etymology
        })

    return parsed_cards

# ==========================================
# 3. Anki 生成
# ==========================================
def generate_anki_package(cards_data, deck_name):
    genanki, tempfile = get_genanki()
    
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2e2e2e; color: #f0f0f0; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; line-height: 1.3; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 20px; text-align: left; }
    .nightMode .definition { color: #e0e0e0; }
    .examples { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 18px; line-height: 1.5; margin-bottom: 15px; text-align: left; }
    .nightMode .examples { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .footer-info { margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 10px; text-align: left; }
    .etymology { display: block; font-size: 16px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; line-height: 1.4; border: 1px solid #fef3c7; }
    .nightMode .etymology { background-color: #333; color: #aaa; border-color: #444; }
    """
    model_id = random.randrange(1 << 30, 1 << 31)
    model = genanki.Model(
        model_id, f'VocabFlow JSON Model {model_id}',
        fields=[{'name': 'FrontPhrase'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Phrase Card',
            'qfmt': '<div class="phrase">{{FrontPhrase}}</div>', 
            'afmt': '''
            {{FrontSide}}<hr>
            <div class="definition">{{Meaning}}</div>
            <div class="examples">{{Examples}}</div>
            {{#Etymology}}
            <div class="footer-info"><div class="etymology">🌱 <b>词源:</b> {{Etymology}}</div></div>
            {{/Etymology}}
            ''',
        }], css=CSS
    )
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    
    for c in cards_data:
        f_phrase = str(c.get('front_phrase', ''))
        f_meaning = str(c.get('meaning', ''))
        f_examples = str(c.get('examples', '')).replace('\n','<br>')
        f_etymology = str(c.get('etymology', ''))
        
        deck.add_note(genanki.Note(model=model, fields=[f_phrase, f_meaning, f_examples, f_etymology]))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 4. Prompt Logic
# ==========================================
def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    
    w_instr = "Key `w`: The word itself (lowercase)." if front_mode == "单词 (Word)" else "Key `w`: A short practical collocation/phrase (2-5 words)."
    m_instr = "Key `m`: Concise Chinese definition." if def_mode == "中文" else "Key `m`: English Definition + Chinese Definition." if def_mode == "中英双语" else "Key `m`: English definition (concise)."
    e_instr = f"Key `e`: {ex_count} example sentence(s). Use `<br>` to separate if multiple."
    r_instr = "Key `r`: Simplified Chinese Etymology (Root/Prefix)." if need_ety else "Key `r`: Leave this empty string \"\"."

    return f"""
Task: Create Anki cards.
Words: {w_list}

**OUTPUT: NDJSON (One line per object).**

**Requirements:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}

**Keys:** `w` (Front), `m` (Meaning), `e` (Examples), `r` (Etymology)

**Example:**
{{"w": "...", "m": "...", "e": "...", "r": "..."}}

**Start:**
"""

# ==========================================
# 5. UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv`")

tab_guide, tab_extract, tab_anki = st.tabs(["📖 使用指南", "1️⃣ 单词提取", "2️⃣ Anki 制作"])

with tab_guide:
    st.markdown("""
    ### 👋 欢迎使用
    **支持格式**：PDF, TXT, EPUB, DOCX，以及 **Kindle 生词本 (.db)**。
    
    <div class="guide-step">
    <span class="guide-title">Step 1: 提取生词</span>
    上传文件（支持 Kindle vocab.db），设置词频过滤，系统会自动提取生词。
    </div>

    <div class="guide-step">
    <span class="guide-title">Step 2: 获取 Prompt</span>
    复制系统生成的 Prompt，发送给 AI (ChatGPT/Claude/DeepSeek)。
    </div>

    <div class="guide-step">
    <span class="guide-title">Step 3: 制作 Anki 牌组</span>
    将 AI 返回的 JSON 粘贴回来，一键生成 .apkg 文件。
    </div>
    """, unsafe_allow_html=True)

with tab_extract:
    mode_context, mode_direct, mode_rank = st.tabs(["📄 语境分析", "📝 直接输入", "🔢 词频列表"])
    
    with mode_context:
        st.info("💡 **全能模式**：支持上传 Kindle `vocab.db`，自动唤醒你的阅读记忆！")
        
        c1, c2 = st.columns(2)
        curr = c1.number_input("忽略排名前 N 的词", 1, 20000, 100, step=100)
        targ = c2.number_input("忽略排名后 N 的词", 2000, 50000, 20000, step=500)
        include_unknown = st.checkbox("🔓 包含生僻词/人名 (Rank > 20000)", value=False)

        # 更新：明确支持 DB 格式
        uploaded_file = st.file_uploader("📂 上传文件 (TXT/PDF/DOCX/EPUB/DB)", key=st.session_state['uploader_id'])
        pasted_text = st.text_area("📄 ...或粘贴文本", height=100, key="paste_key")
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("正在处理...", expanded=True) as status:
                start_time = time.time()
                status.write("📂 读取文件并清洗...")
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                
                # 【新增逻辑】如果提取过程报错返回了 ERROR_ 开头的信息，直接阻断
                if raw_text.startswith("ERROR_"):
                    st.error(raw_text)
                    status.update(label="❌ 解析出错", state="error")
                elif len(raw_text) > 2:
                    status.write("🔍 智能分析与词频比对...")
                    final_data, raw_count, stats_info = analyze_logic(raw_text, curr, targ, include_unknown)
                    
                    if not final_data:
                        st.warning(f"分析完成，但所有 {raw_count} 个单词都被过滤掉了。请尝试调小'忽略排名前 N 的词'或勾选'包含生僻词'。")
                        status.update(label="⚠️ 结果为空", state="error")
                    else:
                        st.session_state['gen_words_data'] = final_data
                        st.session_state['raw_count'] = raw_count
                        st.session_state['stats_info'] = stats_info
                        st.session_state['process_time'] = time.time() - start_time
                        status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容太短或解析为空", state="error")
        
        if st.button("🗑️ 清空", type="secondary", on_click=clear_all_state): pass

    with mode_direct:
        st.info("💡 **直接模式**：不进行词频过滤，直接为粘贴的单词生成 Prompt。")
        raw_input = st.text_area("✍️ 粘贴单词列表 (每行一个 或 逗号分隔)", height=200, placeholder="altruism\nhectic\nserendipity")
        
        if st.button("🚀 生成列表", key="btn_direct", type="primary"):
            if raw_input.strip():
                words = [w.strip() for w in re.split(r'[,\n\t]+', raw_input) if w.strip()]
                seen = set()
                unique_words = []
                for w in words:
                    if w.lower() not in seen:
                        seen.add(w.lower())
                        unique_words.append(w)
                
                data_list = []
                for w in unique_words:
                    rank = VOCAB_DICT.get(w.lower(), 99999) 
                    data_list.append((w, rank))
                
                st.session_state['gen_words_data'] = data_list
                st.session_state['raw_count'] = len(unique_words)
                st.session_state['stats_info'] = None
                
                st.success(f"✅ 已加载 {len(unique_words)} 个单词")
            else:
                st.warning("⚠️ 内容为空")

    with mode_rank:
        gen_type = st.radio("模式", ["🔢 顺序", "🔀 随机"], horizontal=True)
        if "顺序" in gen_type:
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("起始排名", 1, 20000, 1000, step=100)
             count = c_b.number_input("数量", 10, 500, 50, step=10)
             if st.button("🚀 生成"):
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                     data_list = list(zip(subset[w_col], subset[r_col]))
                     st.session_state['gen_words_data'] = data_list
                     st.session_state['raw_count'] = 0
                     st.session_state['stats_info'] = None
        else:
             c_min, c_max, c_cnt = st.columns([1,1,1])
             min_r = c_min.number_input("Min Rank", 1, 20000, 1, step=100)
             max_r = c_max.number_input("Max Rank", 1, 25000, 5000, step=100)
             r_count = c_cnt.number_input("Count", 10, 200, 50, step=10)
             if st.button("🎲 抽取"):
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     mask = (FULL_DF[r_col] >= min_r) & (FULL_DF[r_col] <= max_r)
                     candidates = FULL_DF[mask]
                     if len(candidates) > 0:
                         subset = candidates.sample(n=min(r_count, len(candidates))).sort_values(r_col)
                         data_list = list(zip(subset[w_col], subset[r_col]))
                         st.session_state['gen_words_data'] = data_list
                         st.session_state['raw_count'] = 0
                         st.session_state['stats_info'] = None

    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]
        
        st.divider()
        st.markdown("### 📊 分析报告")
        k1, k2, k3 = st.columns(3)
        raw_c = st.session_state.get('raw_count', 0)
        stats = st.session_state.get('stats_info', {})
        k1.metric("📄 文档总字数", f"{raw_c:,}")
        k2.metric("🎯 筛选生词", f"{len(words_only)}")
        if stats:
            k3.metric("熟词覆盖率", f"{stats.get('coverage', 0):.1%}")
        else:
            k3.metric("熟词覆盖率", "--")
        
        show_rank = st.checkbox("显示单词 Rank", value=False)
        display_text = ", ".join([f"{w}[{r}]" for w, r in data_pairs]) if show_rank else ", ".join(words_only)
            
        with st.expander("📋 **生词预览 (点击展开)**", expanded=False):
            st.markdown(f'<div class="scrollable-text">{display_text}</div>', unsafe_allow_html=True)
            st.code(display_text, language="text")

        with st.expander("⚙️ **自定义 Prompt 设置**", expanded=True):
            col_s1, col_s2 = st.columns(2)
            front_mode = col_s1.selectbox("正面内容", ["短语搭配 (Phrase)", "单词 (Word)"])
            def_mode = col_s2.selectbox("背面释义", ["英文", "中文", "中英双语"])
            col_s3, col_s4 = st.columns(2)
            ex_count = col_s3.slider("例句数量", 1, 3, 1)
            need_ety = col_s4.checkbox("包含词源/词根", value=True)

        batch_size = st.number_input("AI 分组大小", 10, 200, 100, step=10)
        batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📌 第 {idx+1} 组 (共 {len(batch)} 词)", expanded=(idx==0)):
                prompt_text = get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.markdown("👇 **复制 Prompt**")
                st.code(prompt_text, language="text")

with tab_anki:
    st.markdown("### 📦 制作 Anki 牌组")
    if 'anki_cards_cache' not in st.session_state: st.session_state['anki_cards_cache'] = None
    
    def reset_anki_state():
        st.session_state['anki_cards_cache'] = None
        if 'anki_input_text' in st.session_state: st.session_state['anki_input_text'] = ""

    col_input, col_act = st.columns([3, 1])
    with col_input:
        bj_time_str = get_beijing_time_str()
        deck_name = st.text_input("🏷️ 牌组名称", f"Vocab_{bj_time_str}")
    
    st.caption("👇 **粘贴 AI 回复的 JSON** (支持多次追加粘贴)：")
    ai_resp = st.text_area("JSON 输入框", height=300, key="anki_input_text")

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        start_gen = st.button("🚀 开始生成", type="primary", use_container_width=True)
    with c_btn2:
        st.button("🗑️ 清空重置", type="secondary", on_click=reset_anki_state)

    if start_gen or st.session_state['anki_cards_cache'] is not None:
        if start_gen:
            if not ai_resp.strip():
                st.warning("⚠️ 输入框为空")
            else:
                with st.spinner("解析 JSON..."):
                    parsed_data = parse_anki_data(ai_resp)
                    if parsed_data:
                        st.session_state['anki_cards_cache'] = parsed_data
                        st.success(f"✅ 成功提取 {len(parsed_data)} 张卡片")
                    else:
                        st.error("❌ 解析失败，请检查 JSON 格式")
                        st.session_state['anki_cards_cache'] = None

        if st.session_state['anki_cards_cache']:
            cards = st.session_state['anki_cards_cache']
            with st.expander("👀 预览卡片 (Top 50)", expanded=True):
                st.dataframe(pd.DataFrame(cards)[['front_phrase', 'meaning', 'etymology']], use_container_width=True, hide_index=True)

            try:
                f_path = generate_anki_package(cards, deck_name)
                with open(f_path, "rb") as f:
                    st.download_button(f"📥 下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", mime="application/octet-stream", type="primary")
            except Exception as e:
                st.error(f"生成错误: {e}")
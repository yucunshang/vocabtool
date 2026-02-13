import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
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

# 动态 Key 初始化
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
    
    /* 指南样式 (默认浅色) */
    .guide-step { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0056b3; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .guide-title { font-size: 18px; font-weight: bold; color: #0f172a; margin-bottom: 10px; display: block; }
    .guide-tip { font-size: 14px; color: #64748b; background: #eef2ff; padding: 8px; border-radius: 4px; margin-top: 8px; }

    /* 指南样式 (夜间模式适配) */
    @media (prefers-color-scheme: dark) {
        .guide-step { background-color: #262730; border-left: 5px solid #4da6ff; box-shadow: none; border: 1px solid #3d3d3d; border-left: 5px solid #4da6ff; }
        .guide-title { color: #e0e0e0; }
        .guide-tip { background-color: #31333F; color: #b0b0b0; border: 1px solid #444; }
        .scrollable-text { background-color: #262730; border: 1px solid #444; color: #ccc; }
    }
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
    强力清空：
    除了清除分析结果，还会重置文件上传器和文本输入框
    """
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'stats_info']
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    
    if 'paste_key' in st.session_state:
        st.session_state['paste_key'] = ""

# ==========================================
# 2. 核心逻辑 (优化版)
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            bytes_data = uploaded_file.getvalue()
            for encoding in ['utf-8', 'gb18030', 'latin-1']:
                try:
                    text = bytes_data.decode(encoding)
                    break
                except: continue
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
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
        return f"Error: {e}"
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
    """
    V31 优化算法：
    1. 统计阅读覆盖率 (Reading Coverage)
    2. 提取目标生词 (Target Extraction)
    """
    nltk, lemminflect = load_nlp_resources()
    
    def get_lemma_local(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    # 1. 宽松分词
    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_raw_count = len(raw_tokens)
    
    # 2. 统计词频
    valid_tokens = [t.lower() for t in raw_tokens if is_valid_word(t.lower())]
    token_counts = Counter(valid_tokens)
    
    stats_known_count = 0  
    stats_target_count = 0 
    stats_valid_total = sum(token_counts.values()) 
    
    final_candidates = [] 
    seen_lemmas = set()
    
    # 3. 遍历
    for w, count in token_counts.items():
        # A. 计算 Lemma
        lemma = get_lemma_local(w)
        
        # B. 获取 Rank
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        
        if rank_lemma != 99999 and rank_orig != 99999:
            best_rank = min(rank_lemma, rank_orig)
        elif rank_lemma != 99999:
            best_rank = rank_lemma
        else:
            best_rank = rank_orig
            
        # --- 统计逻辑 ---
        if best_rank < current_lvl:
            stats_known_count += count
        elif current_lvl <= best_rank <= target_lvl:
            stats_target_count += count
            
        # --- 提取逻辑 ---
        is_in_range = (best_rank >= current_lvl and best_rank <= target_lvl)
        is_unknown_included = (best_rank == 99999 and include_unknown)
        
        if is_in_range or is_unknown_included:
            word_to_keep = lemma if rank_lemma != 99999 else w
            
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)
    
    # 排序
    final_candidates.sort(key=lambda x: x[1])
    
    # 计算百分比
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
    parsed_cards = []
    text = raw_text.strip()
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'```', '', text).strip()
    
    json_objects = []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            json_objects = data
    except:
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            while pos < len(text) and (text[pos].isspace() or text[pos] == ','):
                pos += 1
            if pos >= len(text):
                break
            try:
                obj, index = decoder.raw_decode(text[pos:])
                json_objects.append(obj)
                pos += index
            except:
                pos += 1

    seen_phrases_lower = set()

    for data in json_objects:
        if not isinstance(data, dict):
            continue
            
        def get_val(keys_list):
            for k in keys_list:
                if k in data: return data[k]
                for data_k in data.keys():
                    if data_k.lower() == k.lower():
                        return data[data_k]
            return ""

        front_text = get_val(['w', 'word', 'phrase', 'term'])
        meaning = get_val(['m', 'meaning', 'def', 'definition'])
        examples = get_val(['e', 'example', 'examples', 'sentence'])
        etymology = get_val(['r', 'root', 'etymology', 'origin'])

        if not front_text or not meaning:
            continue
        
        front_text = str(front_text).replace('**', '').strip()
        meaning = str(meaning).strip()
        examples = str(examples).strip()
        etymology = str(etymology).strip()
        
        if etymology.lower() in ["none", "null", ""]:
            etymology = ""

        if front_text.lower() in seen_phrases_lower: 
            continue
        seen_phrases_lower.add(front_text.lower())

        parsed_cards.append({
            'front_phrase': front_text,
            'meaning': meaning,
            'examples': examples,
            'etymology': etymology
        })

    return parsed_cards

# ==========================================
# 3. Anki 生成 (优化: 字体统一)
# ==========================================
def generate_anki_package(cards_data, deck_name):
    genanki, tempfile = get_genanki()
    
    # 优化 CSS: 例句和词源字体增大到 20px
    CSS = """
    .card { font-family: 'Arial', sans-serif; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2e2e2e; color: #f0f0f0; }
    .phrase { font-size: 28px; font-weight: 700; color: #0056b3; margin-bottom: 20px; line-height: 1.3; }
    .nightMode .phrase { color: #66b0ff; }
    hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0)); margin-bottom: 15px; }
    .definition { font-weight: bold; color: #222; margin-bottom: 15px; font-size: 20px; text-align: left; }
    .nightMode .definition { color: #e0e0e0; }
    .examples { background: #f7f9fa; padding: 12px; border-left: 4px solid #0056b3; border-radius: 4px; color: #444; font-style: italic; font-size: 20px; line-height: 1.5; margin-bottom: 15px; text-align: left; }
    .nightMode .examples { background: #383838; color: #ccc; border-left-color: #66b0ff; }
    .footer-info { margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 10px; text-align: left; }
    .etymology { display: block; font-size: 20px; color: #555; background-color: #fffdf5; padding: 10px; border-radius: 6px; margin-bottom: 5px; line-height: 1.4; border: 1px solid #fef3c7; }
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
        
        deck.add_note(genanki.Note(
            model=model, 
            fields=[f_phrase, f_meaning, f_examples, f_etymology]
        ))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 4. Prompt Logic (优化: 语义原子性)
# ==========================================
def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    
    if front_mode == "单词 (Word)":
        w_instr = "Key `w`: The word itself (lowercase)."
    else:
        w_instr = "Key `w`: A short practical collocation/phrase (2-5 words) that naturally contains the word."

    if def_mode == "中文":
        m_instr = "Key `m`: Concise Chinese definition of the **word** (max 10 chars). NOT the definition of the phrase."
    elif def_mode == "中英双语":
        m_instr = "Key `m`: English Definition + Chinese Definition of the **word**."
    else:
        m_instr = "Key `m`: English definition of the **word** (concise)."

    e_instr = f"Key `e`: {ex_count} example sentence(s). Use `<br>` to separate if multiple."

    if need_ety:
        r_instr = "Key `r`: Simplified Chinese Etymology (Root/Prefix) corresponding to this specific meaning."
    else:
        r_instr = "Key `r`: Leave this empty string \"\"."

    return f"""
Task: Create Anki cards.
Words: {w_list}

**CRITICAL: SEMANTIC ATOMICITY**
1. **Consistency**: The Word/Phrase (`w`), Meaning (`m`), Example (`e`), and Etymology (`r`) MUST all correspond to the **same specific context/meaning**.
2. **No Mixing**: Do NOT mix definitions. (e.g., If `w` is "bracket" in a tax context, `m` must be "grade/category", `e` must be about taxes. Do NOT give the definition of "punctuation mark").
3. **Definition Focus**: Even if `w` is a phrase (e.g. "give up"), `m` should explain the core meaning derived from it.

**Output Format: NDJSON (One line per object).**

**Requirements:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}

**Keys:** `w` (Front), `m` (Meaning), `e` (Examples), `r` (Etymology)

**Example (Correct Consistency):**
{{"w": "bracket", "m": "等级/档次", "e": "He is in the highest income tax bracket.", "r": "from braguette (codpiece)"}}

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
    ### 👋 欢迎使用 Vocab Flow Ultra
    这是一个**从阅读材料中提取生词**，并利用 **AI** 自动生成 **Anki 卡片**的效率工具。
    
    ---
    
    <div class="guide-step">
    <span class="guide-title">Step 1: 提取生词 (Extract)</span>
    在 <code>1️⃣ 单词提取</code> 标签页：<br><br>
    <strong>1. 上传文件</strong><br>
    支持 PDF, TXT, EPUB, DOCX。无论是小说、文章还是单词表，直接丢进去即可。<br>
    系统会自动进行 <strong>NLP 词形还原</strong>（将 went 还原为 go）并清洗垃圾词（乱码、重复字符）。<br>
    <br>
    <strong>2. 设置过滤范围 (Rank Filter)</strong><br>
    利用 COCA 20000 词频表进行科学筛选：
    <ul>
        <li><strong>忽略排名前 N</strong> (Min Rank)：例如设为 <code>6000</code>，会过滤掉基础词汇。</li>
        <li><strong>忽略排名后 N</strong> (Max Rank)：例如设为 <code>10000</code>，专注于进阶词汇。</li>
        <li><strong>🔓 包含生僻词</strong> (Unknown)：勾选后，将强制包含词频表中没有的词（如人名、地名、新造词）。</li>
    </ul>
    <br>
    <strong>3. 点击 🚀 开始分析</strong><br>
    系统会融合处理，自动去重并按词频排序，最大化提取有效单词。
    </div>

    <div class="guide-step">
    <span class="guide-title">Step 2: 获取 Prompt (AI Generation)</span>
    分析完成后：<br><br>
    <strong>1. 自定义设置</strong><br>
    点击 <code>⚙️ 自定义 Prompt 设置</code>，选择正面是单词还是短语，释义语言等。<br>
    <br>
    <strong>2. 复制 Prompt</strong><br>
    系统会自动将单词分组。生成的单词表支持<strong>折叠</strong>和<strong>滚动查看</strong>。<br>
    <ul>
        <li>📱 <strong>手机/鸿蒙端</strong>：使用下方的“纯文本框”，长按全选 -> 复制。</li>
        <li>💻 <strong>电脑端</strong>：点击代码块右上角的 Copy 📄 图标。</li>
    </ul>
    <br>
    <strong>3. 发送给 AI</strong><br>
    将复制的内容发送给 ChatGPT / Claude / Gemini / DeepSeek。AI 会返回一串 JSON 数据。
    </div>

    <div class="guide-step">
    <span class="guide-title">Step 3: 制作 Anki 牌组 (Create Deck)</span>
    在 <code>2️⃣ Anki 制作</code> 标签页：<br><br>
    <strong>1. 粘贴 AI 回复</strong><br>
    将 AI 生成的 JSON 内容粘贴到输入框中。<br>
    <div class="guide-tip">💡 <strong>支持追加粘贴</strong>：如果你有 5 组单词，可以把 AI 的 5 次回复依次粘贴在同一个框里，不需要分批下载。</div>
    <br>
    <strong>2. 点击“开始生成”</strong><br>
    粘贴完所有内容后，点击生成按钮，系统将解析 JSON 并生成文件。
    <br>
    <strong>3. 下载与导入</strong><br>
    点击 <strong>📥 下载 .apkg</strong>，然后双击该文件，它会自动导入到你的 Anki 软件中。
    </div>
    """, unsafe_allow_html=True)

with tab_extract:
    mode_context, mode_rank = st.tabs(["📄 语境分析", "🔢 词频列表"])
    
    with mode_context:
        # V29: 统一模式，只保留筛选器
        st.info("💡 **全能模式**：系统将自动进行 NLP 词形还原、去重、垃圾词清洗。无论是文章还是单词表，直接上传即可。")
        
        c1, c2 = st.columns(2)
        # 默认值修改：6000 / 10000
        curr = c1.number_input("忽略排名前 N 的词", 1, 20000, 6000, step=100)
        targ = c2.number_input("忽略排名后 N 的词", 2000, 50000, 10000, step=500)
        include_unknown = st.checkbox("🔓 包含生僻词/人名 (Rank > 20000)", value=False)

        uploaded_file = st.file_uploader("📂 上传文档 (TXT/PDF/DOCX/EPUB)", key=st.session_state['uploader_id'])
        pasted_text = st.text_area("📄 ...或粘贴文本", height=100, key="paste_key")
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("正在处理...", expanded=True) as status:
                start_time = time.time()
                status.write("📂 读取文件并清洗垃圾词...")
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                
                if len(raw_text) > 2:
                    status.write("🔍 智能分析、计算阅读覆盖率...")
                    
                    # 调用新版逻辑，解包返回值
                    final_data, raw_count, stats_info = analyze_logic(raw_text, curr, targ, include_unknown)
                    
                    st.session_state['gen_words_data'] = final_data # [(word, rank), ...]
                    st.session_state['raw_count'] = raw_count
                    st.session_state['stats_info'] = stats_info
                    st.session_state['process_time'] = time.time() - start_time
                    
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 内容太短", state="error")
        
        if st.button("🗑️ 清空", type="secondary", on_click=clear_all_state): pass

    with mode_rank:
        gen_type = st.radio("模式", ["🔢 顺序", "🔀 随机"], horizontal=True)
        if "顺序" in gen_type:
             c_a, c_b = st.columns(2)
             s_rank = c_a.number_input("起始排名", 1, 20000, 1000, step=100)
             # V32: 上限调整为 5000
             count = c_b.number_input("数量", 10, 5000, 50, step=50)
             if st.button("🚀 生成"):
                 start_time = time.time()
                 if FULL_DF is not None:
                     r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                     w_col = next(c for c in FULL_DF.columns if 'word' in c)
                     subset = FULL_DF[FULL_DF[r_col] >= s_rank].sort_values(r_col).head(count)
                     data_list = list(zip(subset[w_col], subset[r_col]))
                     st.session_state['gen_words_data'] = data_list
                     st.session_state['raw_count'] = 0
                     st.session_state['stats_info'] = None
                     st.session_state['process_time'] = time.time() - start_time
        else:
             c_min, c_max, c_cnt = st.columns([1,1,1])
             min_r = c_min.number_input("Min Rank", 1, 20000, 1, step=100)
             max_r = c_max.number_input("Max Rank", 1, 25000, 5000, step=100)
             # V32: 上限调整为 5000
             r_count = c_cnt.number_input("Count", 10, 5000, 50, step=50)
             if st.button("🎲 抽取"):
                 start_time = time.time()
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
                         st.session_state['process_time'] = time.time() - start_time

    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        # 解包数据
        data_pairs = st.session_state['gen_words_data']
        words_only = [p[0] for p in data_pairs]
        
        st.divider()
        st.markdown("### 📊 分析报告")
        
        # 显示 4 个指标
        k1, k2, k3, k4 = st.columns(4)
        raw_c = st.session_state.get('raw_count', 0)
        p_time = st.session_state.get('process_time', 0.1)
        stats = st.session_state.get('stats_info', {})
        
        k1.metric("📄 总字数", f"{raw_c:,}")
        
        if stats:
            k2.metric("📖 熟词覆盖率", f"{stats.get('coverage', 0):.1%}")
            k3.metric("🎯 重点词占比", f"{stats.get('target_density', 0):.1%}")
        else:
            k2.metric("📖 熟词覆盖率", "--")
            k3.metric("🎯 重点词占比", "--")

        k4.metric("📝 提取生词", f"{len(words_only)}")
        
        # --- V29: 增强版预览区 (折叠+Rank) ---
        show_rank = st.checkbox("显示单词 Rank", value=False)
        
        if show_rank:
            display_text = ", ".join([f"{w}[{r}]" for w, r in data_pairs])
        else:
            display_text = ", ".join(words_only)
            
        with st.expander("📋 **全部生词预览 (点击展开/折叠)**", expanded=False):
            st.markdown(f'<div class="scrollable-text">{display_text}</div>', unsafe_allow_html=True)
            st.caption("提示：长按上方文本框可全选复制，或点击下方代码块复制按钮。")
            st.code(display_text, language="text")

        with st.expander("⚙️ **自定义 Prompt 设置 (点击展开)**", expanded=True):
            col_s1, col_s2 = st.columns(2)
            front_mode = col_s1.selectbox("正面内容", ["短语搭配 (Phrase)", "单词 (Word)"])
            def_mode = col_s2.selectbox("背面释义", ["英文", "中文", "中英双语"])
            
            col_s3, col_s4 = st.columns(2)
            ex_count = col_s3.slider("例句数量", 1, 3, 1)
            need_ety = col_s4.checkbox("包含词源/词根", value=True)

        # 默认 Batch Size 修改为 150
        batch_size = st.number_input("AI 分组大小", 50, 500, 150, step=10)
        batches = [words_only[i:i + batch_size] for i in range(0, len(words_only), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📌 第 {idx+1} 组 (共 {len(batch)} 词)", expanded=(idx==0)):
                prompt_text = get_ai_prompt(batch, front_mode, def_mode, ex_count, need_ety)
                st.caption("📱 手机端专用：")
                st.text_area(f"text_area_{idx}", value=prompt_text, height=100, label_visibility="collapsed")
                st.caption("💻 电脑端：")
                st.code(prompt_text, language="text")

with tab_anki:
    st.markdown("### 📦 制作 Anki 牌组")
    
    # --- 状态初始化 ---
    if 'anki_cards_cache' not in st.session_state:
        st.session_state['anki_cards_cache'] = None
    
    def reset_anki_state():
        st.session_state['anki_cards_cache'] = None
        if 'anki_input_text' in st.session_state:
             st.session_state['anki_input_text'] = ""

    # --- 1. 设置区域 ---
    col_input, col_act = st.columns([3, 1])
    
    with col_input:
        bj_time_str = get_beijing_time_str()
        deck_name = st.text_input("🏷️ 牌组名称 (Deck Name)", f"Vocab_{bj_time_str}", help="导入 Anki 后显示的牌组名字")
    
    st.caption("👇 **在此粘贴 AI 回复的 JSON 数据** (支持多次追加粘贴，粘贴完所有内容后点击生成)：")
    
    # 绑定 session_state key，这样输入内容不会轻易丢失
    ai_resp = st.text_area(
        "JSON 输入框", 
        height=300, 
        key="anki_input_text",
        placeholder='''[
  {"w": "serendipity", "m": "意外发现珍奇事物的本领", "e": "It was pure serendipity...", "r": "coined by Horace Walpole"},
  ...
]'''
    )

    # --- 2. 操作按钮区 ---
    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        # 核心改动：只有点击这个按钮才开始解析
        start_gen = st.button("🚀 开始生成", type="primary", use_container_width=True)
    with c_btn2:
        st.button("🗑️ 清空重置", type="secondary", on_click=reset_anki_state)

    # --- 3. 逻辑处理 ---
    # 如果点击了生成按钮，或者缓存里已经有数据（处理下载按钮刷新问题）
    if start_gen or st.session_state['anki_cards_cache'] is not None:
        
        # 如果是点击了按钮，进行解析
        if start_gen:
            if not ai_resp.strip():
                st.warning("⚠️ 输入框为空，请先粘贴 AI 生成的 JSON 内容。")
            else:
                with st.spinner("正在解析 JSON 并构建卡片..."):
                    parsed_data = parse_anki_data(ai_resp)
                    if parsed_data:
                        st.session_state['anki_cards_cache'] = parsed_data
                        st.success(f"✅ 成功提取 {len(parsed_data)} 张卡片！")
                    else:
                        st.error("❌ 解析失败：未找到有效的 JSON 数据。请检查是否包含了完整的 `{...}` 结构。")
                        st.session_state['anki_cards_cache'] = None

        # --- 4. 结果展示与下载 ---
        # 只要缓存有数据就显示，独立于按钮点击状态
        if st.session_state['anki_cards_cache']:
            cards = st.session_state['anki_cards_cache']
            
            # 预览表格
            with st.expander("👀 预览卡片内容 (Top 50)", expanded=True):
                df_view = pd.DataFrame(cards)
                # 简单重命名以便预览友好
                df_preview = df_view.rename(columns={
                    'front_phrase': '正面 (Front)', 
                    'meaning': '背面 (Back)', 
                    'examples': '例句', 
                    'etymology': '词源'
                })
                st.dataframe(df_preview, use_container_width=True, hide_index=True)

            # 生成文件 (每次渲染时生成，确保 Deck Name 是最新的)
            try:
                f_path = generate_anki_package(cards, deck_name)
                
                # 下载按钮
                with open(f_path, "rb") as f:
                    file_data = f.read()
                    
                st.download_button(
                    label=f"📥 下载 {deck_name}.apkg",
                    data=file_data,
                    file_name=f"{deck_name}.apkg",
                    mime="application/octet-stream",
                    type="primary",
                    help="点击下载后，双击文件即可导入 Anki"
                )
            except Exception as e:
                st.error(f"生成 .apkg 文件时发生错误: {e}")
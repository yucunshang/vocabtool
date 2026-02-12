import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra V31", 
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
    .scrollable-text {
        max-height: 200px; overflow-y: auto; padding: 10px;
        border: 1px solid #eee; border-radius: 5px; background-color: #fafafa;
        font-family: monospace; white-space: pre-wrap;
    }
    .guide-step { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #0056b3; }
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
        # 增加 'punkt' 用于分句
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
    return (utc_now + timedelta(hours=8)).strftime('%Y-%m-%d') # 改为日期格式用于Tag

def clear_all_state():
    keys_to_drop = ['gen_words_data', 'raw_count', 'process_time', 'raw_text_input']
    for k in keys_to_drop:
        if k in st.session_state: del st.session_state[k]
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    if 'paste_key' in st.session_state: st.session_state['paste_key'] = ""
    if 'anki_input_text' in st.session_state: st.session_state['anki_input_text'] = ""

# ==========================================
# 2. 核心逻辑 (含语境提取)
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
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
    except Exception as e: return f"Error: {e}"
    return text

def is_valid_word(word):
    if len(word) < 2 or len(word) > 25: return False
    if re.search(r'(.)\1{2,}', word): return False # 连续3个相同字母
    if not re.search(r'[aeiouy]', word): return False # 无元音
    return True

# --- NEW: 语境查找函数 (轻量级) ---
def find_context_sentence(text, target_word, nltk_instance):
    """
    在全文中找到包含目标词的第一个句子。
    限制长度，防止找到几千字的超长段落。
    """
    sentences = nltk_instance.sent_tokenize(text)
    # 简单的正则边界匹配，防止 matching "us" in "virus"
    pattern = re.compile(r'\b' + re.escape(target_word) + r'\b', re.IGNORECASE)
    
    for sent in sentences:
        if len(sent) > 300: continue # 跳过过长的句子(可能是解析错误)
        if pattern.search(sent):
            # 清洗一下空白符
            return re.sub(r'\s+', ' ', sent).strip()
    return ""

def analyze_logic(text, current_lvl, target_lvl, include_unknown):
    nltk, lemminflect = load_nlp_resources()
    
    def get_lemma_local(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_words = len(raw_tokens)
    clean_tokens = set([t.lower() for t in raw_tokens if is_valid_word(t.lower())])
    
    final_candidates = [] 
    seen_lemmas = set()
    
    for w in clean_tokens:
        lemma = get_lemma_local(w)
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        
        best_rank = rank_lemma if rank_lemma != 99999 else rank_orig
        if rank_lemma != 99999 and rank_orig != 99999: best_rank = min(rank_lemma, rank_orig)

        is_in_range = (best_rank >= current_lvl and best_rank <= target_lvl)
        is_unknown = (best_rank == 99999 and include_unknown)
        
        if is_in_range or is_unknown:
            word_to_keep = lemma if rank_lemma != 99999 else w
            if lemma not in seen_lemmas:
                # --- NEW: 这里只存基本信息，语境等展示时再提取，节省内存 ---
                final_candidates.append({
                    "word": word_to_keep,
                    "rank": best_rank,
                    "orig_word": w # 存一下原词，方便找例句
                })
                seen_lemmas.add(lemma)
    
    final_candidates.sort(key=lambda x: x["rank"])
    
    # --- NEW: 只有在最终列表确定后，才去提取语境 (性能优化) ---
    # 限制最大处理数量，防止 text 太大卡死
    for item in final_candidates[:500]: 
        ctx = find_context_sentence(text, item["orig_word"], nltk)
        item["context"] = ctx

    return final_candidates, total_words

# ==========================================
# 3. 数据解析与 Anki (含编辑功能)
# ==========================================
def parse_anki_data(raw_text):
    text = raw_text.strip()
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'```', '', text).strip()
    
    json_objects = []
    try:
        data = json.loads(text)
        if isinstance(data, list): json_objects = data
    except:
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            while pos < len(text) and (text[pos].isspace() or text[pos] == ','): pos += 1
            if pos >= len(text): break
            try:
                obj, index = decoder.raw_decode(text[pos:])
                json_objects.append(obj)
                pos += index
            except: pos += 1

    parsed_cards = []
    seen = set()
    for data in json_objects:
        if not isinstance(data, dict): continue
        
        def get_val(keys):
            for k in keys:
                if k in data: return data[k]
                for dk in data.keys():
                    if dk.lower() == k: return data[dk]
            return ""

        f = str(get_val(['w', 'word', 'phrase'])).replace('**', '').strip()
        m = str(get_val(['m', 'meaning', 'def'])).strip()
        e = str(get_val(['e', 'example', 'sentence'])).strip()
        r = str(get_val(['r', 'root', 'etymology'])).strip()
        
        if not f or not m: continue
        if r.lower() in ["none", "null", ""]: r = ""
        
        if f.lower() not in seen:
            parsed_cards.append({'正面': f, '背面': m, '例句': e, '词源': r})
            seen.add(f.lower())
    return parsed_cards

def generate_anki_package(df_data, deck_name):
    genanki, tempfile = get_genanki()
    
    # CSS 优化：高亮词源，更好的排版
    CSS = """
    .card { font-family: arial; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .phrase { font-size: 28px; font-weight: bold; color: #0056b3; margin-bottom: 15px; }
    .definition { color: #333; margin-bottom: 15px; font-size: 20px; text-align: left; }
    .examples { background: #f7f9fa; padding: 10px; border-left: 3px solid #0056b3; color: #555; font-style: italic; font-size: 18px; text-align: left; }
    .etymology { margin-top: 15px; padding: 8px; border: 1px dashed #ccc; border-radius: 5px; background: #fffdf5; color: #666; font-size: 16px; text-align: left; }
    .highlight { color: #d97706; font-weight: bold; }
    """
    
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31),
        'VocabFlow V31 Model',
        fields=[{'name': 'Front'}, {'name': 'Back'}, {'name': 'Example'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="phrase">{{Front}}</div>',
            'afmt': '{{FrontSide}}<hr><div class="definition">{{Back}}</div><div class="examples">{{Example}}</div>{{#Etymology}}<div class="etymology">🌱 <b>词源:</b> {{Etymology}}</div>{{/Etymology}}',
        }], css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    date_tag = get_beijing_time_str()
    
    for _, row in df_data.iterrows():
        note = genanki.Note(
            model=model,
            fields=[str(row['正面']), str(row['背面']), str(row['例句']).replace('\n','<br>'), str(row['词源'])],
            tags=['VocabFlow', f'Date_{date_tag}'] # 自动打标签
        )
        deck.add_note(note)
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 4. Prompt Logic (含语境)
# ==========================================
def get_ai_prompt(data_batch, front_mode, def_mode, ex_count, need_ety):
    # data_batch 是 [{'word': '...', 'context': '...'}, ...]
    
    input_str = ""
    for item in data_batch:
        ctx_str = f" (Context: {item['context']})" if item.get('context') else ""
        input_str += f"- {item['word']}{ctx_str}\n"

    w_instr = "Key `w`: The word/phrase."
    m_instr = f"Key `m`: {def_mode} definition fitting the Context."
    e_instr = f"Key `e`: {ex_count} example(s). **IMPORTANT: The first example MUST be the provided 'Context' sentence (if available).**"
    r_instr = "Key `r`: Etymology (Root/Affix breakdown) e.g. 're-(again)+flect(bend)'." if need_ety else "Key `r`: Empty string."

    return f"""
Task: Create Anki cards for language learning.
**INPUT WORDS & CONTEXT:**
{input_str}

**OUTPUT FORMAT: JSON List**
[
  {{"w": "word", "m": "definition", "e": "examples", "r": "etymology"}}
]

**REQUIREMENTS:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}
"""

# ==========================================
# 5. UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow Ultra V31")

if not VOCAB_DICT: st.error("⚠️ 缺失 `coca_cleaned.csv`")

tab_guide, tab_extract, tab_anki = st.tabs(["📖 指南", "1️⃣ 提取 & Prompt", "2️⃣ 制作 Anki"])

with tab_guide:
    st.markdown("""
    ### 🚀 V31 新特性
    1. **语境感知**：提取单词时会自动抓取**原句**，Prompt 会指示 AI 将其作为第一个例句。
    2. **完全掌控**：在生成 Anki 包之前，你可以像操作 Excel 一样**修改表格数据**。
    3. **自动标签**：导入 Anki 后，卡片会自动带有 `VocabFlow` 和 `Date_YYYY-MM-DD` 标签。
    """)

with tab_extract:
    c1, c2 = st.columns(2)
    curr = c1.number_input("Min Rank (忽略高频)", 1, 20000, 100, step=100)
    targ = c2.number_input("Max Rank (忽略生僻)", 2000, 50000, 20000, step=500)
    include_unknown = st.checkbox("包含未知词 (Rank > 20000)", False)

    uploaded_file = st.file_uploader("📂 上传文档", key=st.session_state['uploader_id'])
    pasted_text = st.text_area("...或粘贴文本", height=100, key="paste_key")
    
    if st.button("🚀 开始分析", type="primary"):
        with st.status("处理中...", expanded=True) as status:
            raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
            if len(raw_text) > 2:
                # 存入 Session State 方便后续使用
                st.session_state['raw_text_input'] = raw_text 
                final_data, raw_count = analyze_logic(raw_text, curr, targ, include_unknown)
                st.session_state['gen_words_data'] = final_data
                st.session_state['raw_count'] = raw_count
                status.update(label="✅ 完成", state="complete", expanded=False)
            else:
                status.update(label="⚠️ 内容太短", state="error")

    if 'gen_words_data' in st.session_state:
        data = st.session_state['gen_words_data'] # List of dicts
        words_only = [d['word'] for d in data]
        
        st.divider()
        st.metric("🎯 筛选生词", f"{len(data)} (原文总字数: {st.session_state.get('raw_count',0)})")
        
        with st.expander("📋 生词预览 (带语境)", expanded=False):
            # 简单的文本展示
            preview_txt = "\n".join([f"{d['word']} [{d['rank']}] -> {d.get('context','(无语境)')[:50]}..." for d in data])
            st.text(preview_txt)

        with st.expander("⚙️ Prompt 设置", expanded=True):
            c_p1, c_p2 = st.columns(2)
            def_mode = c_p1.selectbox("释义语言", ["英文", "中文", "中英双语"])
            need_ety = c_p2.checkbox("包含词源拆解", True)
        
        batch_size = st.number_input("分组大小", 10, 200, 50)
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        for idx, batch in enumerate(batches):
            with st.expander(f"📄 Prompt 第 {idx+1} 组"):
                p_text = get_ai_prompt(batch, "Word", def_mode, 1, need_ety)
                st.code(p_text, language="text")

with tab_anki:
    st.markdown("### 🛠️ 编辑与生成")
    
    ai_resp = st.text_area("👇 粘贴 AI 返回的 JSON", height=150, key="anki_input_text", placeholder='[{"w": "...", "m": "..."}]')
    
    # Session State 用于存储解析后的 DataFrame，防止每次刷新都重置
    if 'df_preview' not in st.session_state: st.session_state['df_preview'] = None

    if st.button("🔍 解析预览"):
        if ai_resp.strip():
            parsed = parse_anki_data(ai_resp)
            if parsed:
                st.session_state['df_preview'] = pd.DataFrame(parsed)
                st.success(f"解析成功，共 {len(parsed)} 条。请在下方表格确认内容。")
            else:
                st.error("解析失败，请检查 JSON 格式。")
    
    # --- NEW: 可编辑表格 (Human-in-the-loop) ---
    if st.session_state['df_preview'] is not None:
        st.markdown("✏️ **请在下方直接修改内容，确认无误后下载**")
        edited_df = st.data_editor(st.session_state['df_preview'], num_rows="dynamic", use_container_width=True)
        
        c_d1, c_d2 = st.columns([2, 1])
        deck_name = c_d1.text_input("牌组名", f"Vocab_{get_beijing_time_str()}")
        
        if c_d2.button("📥 生成 .apkg", type="primary"):
            if not edited_df.empty:
                f_path = generate_anki_package(edited_df, deck_name)
                with open(f_path, "rb") as f:
                    st.download_button(f"点击下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", mime="application/octet-stream")
            else:
                st.warning("表格为空！")
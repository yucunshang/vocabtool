import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 页面配置 & 极速重置逻辑
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Fast", 
    page_icon="⚡️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# 核心重置函数：瞬间清空所有状态并刷新
def reset_app():
    # 1. 保留部分无需重置的配置(如果有)，这里选择全清
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # 2. 重新生成上传组件的 ID，强制 UI 丢弃旧文件
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    
    # 3. 强制重新运行脚本 (比 F5 快很多)
    st.rerun()

# 初始化上传器 ID
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"

# 侧边栏添加重置按钮
with st.sidebar:
    st.header("功能菜单")
    if st.button("🔄 一键重置 / 清空", type="primary"):
        reset_app()
    st.caption("点击此按钮可瞬间清空所有内容并重置系统。")

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; }
    .scrollable-text {
        max-height: 200px; overflow-y: auto; padding: 10px;
        border: 1px solid #eee; border-radius: 5px;
        background-color: #fafafa; font-family: monospace; white-space: pre-wrap;
    }
    /* 侧边栏按钮样式优化 */
    [data-testid="stSidebar"] button {
        background-color: #ff4b4b;
        color: white;
        border: none;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #ff2b2b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载 (带缓存)
# ==========================================
@st.cache_resource(show_spinner="加载 NLP 引擎...")
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
            except LookupError: 
                try: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
                except: pass
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
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

# ==========================================
# 2. 核心逻辑
# ==========================================
def extract_text_from_file(uploaded_file):
    pypdf, docx, ebooklib, epub, BeautifulSoup = get_file_parsers()
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'txt':
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
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
    if re.search(r'(.)\1{2,}', word): return False
    if not re.search(r'[aeiouy]', word): return False
    return True

def analyze_logic(text, current_lvl, target_lvl, include_unknown):
    nltk, lemminflect = load_nlp_resources()
    def get_lemma_local(word):
        try: return lemminflect.getLemma(word, upos='VERB')[0]
        except: return word

    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    clean_tokens = set([t.lower() for t in raw_tokens if is_valid_word(t.lower())])
    final_candidates = []
    seen_lemmas = set()
    
    for w in clean_tokens:
        lemma = get_lemma_local(w)
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        best_rank = min(rank_lemma, rank_orig) if rank_lemma!=99999 and rank_orig!=99999 else (rank_lemma if rank_lemma!=99999 else rank_orig)
        
        if (best_rank >= current_lvl and best_rank <= target_lvl) or (best_rank == 99999 and include_unknown):
            word_to_keep = lemma if rank_lemma != 99999 else w
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)
    
    final_candidates.sort(key=lambda x: x[1])
    return final_candidates, len(raw_tokens)

def parse_anki_data(raw_text):
    parsed_cards = []
    text = raw_text.replace("```json", "").replace("```", "").strip()
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    seen_phrases = set()

    for match in matches:
        try:
            data = json.loads(match.group(), strict=False)
            f, m, e, r = data.get("w"), data.get("m"), data.get("e"), data.get("r", "")
            if not f or not m: continue
            if f.lower() in seen_phrases: continue
            seen_phrases.add(f.lower())
            parsed_cards.append({'front_phrase': f, 'meaning': m, 'examples': e, 'etymology': r})
        except: continue
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
        templates=[{'name': 'Card 1', 'qfmt': '<div class="phrase">{{FrontPhrase}}</div>', 'afmt': '{{FrontSide}}<hr><div class="definition">{{Meaning}}</div><div class="examples">{{Examples}}</div>{{#Etymology}}<div class="footer-info"><div class="etymology">🌱 <b>词源:</b> {{Etymology}}</div></div>{{/Etymology}}'}], css=CSS)
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[str(c['front_phrase']), str(c['meaning']), str(c['examples']).replace('\n','<br>'), str(c['etymology'])]))
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

# ==========================================
# 4. Prompt Logic (修复完整)
# ==========================================
def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    w_instr = "Key `w`: The word itself (lowercase)." if front_mode == "单词 (Word)" else "Key `w`: A short practical collocation/phrase (2-5 words)."
    m_instr = "Key `m`: Concise Chinese definition." if def_mode == "中文" else ("Key `m`: English + Chinese Definition." if def_mode == "中英双语" else "Key `m`: English definition.")
    e_instr = f"Key `e`: {ex_count} example sentence(s). Use `<br>` to separate if multiple."
    r_instr = "Key `r`: Simplified Chinese Etymology." if need_ety else "Key `r`: Empty string \"\"."

    return f"""
Task: Create Anki cards.
Words: {w_list}

**OUTPUT: NDJSON (One line per object).**

**CRITICAL RULES:**
1. **NO unescaped double quotes** inside values. 
   - WRONG: "meaning": "It means "hello""
   - RIGHT: "meaning": "It means \\"hello\\"" (Use backslash)
   - RIGHT: "meaning": "It means 'hello'" (Use single quotes)

**Fields:**
1. {w_instr}
2. {m_instr}
3. {e_instr}
4. {r_instr}

**Format Example:**
{{"w": "...", "m": "...", "e": "...", "r": "..."}}

**Start:**
"""

# ==========================================
# 5. UI 主程序
# ==========================================
st.title("⚡️ Vocab Flow Fast")

if not VOCAB_DICT: st.error("⚠️ 缺失 `coca_cleaned.csv`")

tab_extract, tab_anki = st.tabs(["1️⃣ 单词提取", "2️⃣ Anki 制作"])

with tab_extract:
    mode_context, mode_rank = st.tabs(["📄 语境分析", "🔢 词频列表"])
    with mode_context:
        st.info("💡 稳定版默认：Rank 8000-15000")
        c1, c2 = st.columns(2)
        curr = c1.number_input("Min Rank", 1, 20000, 8000, step=500)
        targ = c2.number_input("Max Rank", 2000, 50000, 15000, step=500)
        include_unknown = st.checkbox("🔓 包含生僻词", value=False)
        uploaded_file = st.file_uploader("📂 上传文件", key=st.session_state['uploader_id'])
        pasted_text = st.text_area("📄 粘贴文本", height=100, key="paste_key")
        
        if st.button("🚀 开始分析", type="primary"):
            with st.status("处理中...", expanded=True) as status:
                st.session_state['process_time'] = time.time()
                raw_text = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
                if len(raw_text) > 2:
                    final_data, raw_count = analyze_logic(raw_text, curr, targ, include_unknown)
                    st.session_state['gen_words_data'] = final_data
                    st.session_state['raw_count'] = raw_count
                    st.session_state['process_time'] = time.time() - st.session_state['process_time']
                    status.update(label="✅ 完成", state="complete", expanded=False)
                else: status.update(label="⚠️ 内容太短", state="error")
        if st.button("🗑️ 清空当前", type="secondary", on_click=reset_app): pass

    with mode_rank:
        c_min, c_max, c_cnt = st.columns([1,1,1])
        min_r = c_min.number_input("Min", 1, 20000, 8000)
        max_r = c_max.number_input("Max", 1, 25000, 15000)
        r_count = c_cnt.number_input("Count", 10, 500, 50)
        if st.button("🎲 抽取"):
             if FULL_DF is not None:
                 r_col, w_col = next(c for c in FULL_DF.columns if 'rank' in c), next(c for c in FULL_DF.columns if 'word' in c)
                 subset = FULL_DF[(FULL_DF[r_col] >= min_r) & (FULL_DF[r_col] <= max_r)]
                 if len(subset) > 0:
                     sample = subset.sample(n=min(r_count, len(subset))).sort_values(r_col)
                     st.session_state['gen_words_data'] = list(zip(sample[w_col], sample[r_col]))
                     st.session_state['raw_count'] = 0

    if 'gen_words_data' in st.session_state and st.session_state['gen_words_data']:
        data = st.session_state['gen_words_data']
        words = [p[0] for p in data]
        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.metric("字数", st.session_state.get('raw_count', 0))
        k2.metric("生词", len(words))
        k3.metric("耗时", f"{st.session_state.get('process_time', 0):.2f}s")
        
        with st.expander("📋 生词预览", expanded=False):
            txt = ", ".join([f"{w}[{r}]" for w,r in data])
            st.markdown(f'<div class="scrollable-text">{txt}</div>', unsafe_allow_html=True)
            st.code(txt)

        with st.expander("⚙️ 设置", expanded=True):
            c1, c2 = st.columns(2)
            fm = c1.selectbox("正面", ["短语 (Phrase)", "单词 (Word)"])
            dm = c2.selectbox("背面", ["英文", "中文", "中英双语"])
            c3, c4 = st.columns(2)
            ec = c3.slider("例句", 1, 3, 1)
            ne = c4.checkbox("词源", value=True)

        bs = st.number_input("分组大小", 10, 500, 100)
        for i, batch in enumerate([words[i:i+bs] for i in range(0, len(words), bs)]):
            with st.expander(f"📌 第 {i+1} 组", expanded=(i==0)):
                st.code(get_ai_prompt(batch, fm, dm, ec, ne), language="text")

with tab_anki:
    st.markdown("### 📦 制作 Anki")
    ai_resp = st.text_area("JSON 输入", height=300, key="anki_input_text")
    d_name = st.text_input("牌组名", f"Vocab_{get_beijing_time_str()}")
    if ai_resp.strip():
        parsed = parse_anki_data(ai_resp)
        if parsed:
            st.success(f"✅ 解析 {len(parsed)} 条")
            st.download_button(f"📥 下载 .apkg", open(generate_anki_package(parsed, d_name), "rb"), file_name=f"{d_name}.apkg", mime="application/octet-stream", type="primary")
        else: st.warning("⚠️ 等待粘贴...")
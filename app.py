# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re
import os
import random
import json
import time
from datetime import datetime, timedelta, timezone

# ==========================================
# 0. 页面配置 (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra (CN-Stable)",
    page_icon="⚡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 注入自定义 CSS (优化中文显示与排版)
st.markdown("""
<style>
    /* 字体优化 */
    .stTextArea textarea { font-family: 'Consolas', 'Courier New', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; margin-top: 5px; }
    
    /* 滚动文本框样式 */
    .scrollable-text {
        max-height: 250px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 5px;
        background-color: #fafafa;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 13px;
        color: #333;
    }
    
    /* 指南卡片样式 */
    .guide-step { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 15px; 
        border-left: 4px solid #0056b3; 
    }
    .guide-title { 
        font-weight: bold; 
        color: #0f172a; 
        display: block; 
        margin-bottom: 5px; 
        font-size: 16px;
    }
    
    /* 针对网络加载慢的提示框 */
    .network-warning {
        padding: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        border-radius: 5px;
        margin-bottom: 10px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session State
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = "1000"

# ==========================================
# 1. 核心资源加载 (Network Robustness)
# ==========================================

@st.cache_resource(show_spinner="正在初始化 NLP 引擎...")
def load_nlp_resources():
    """
    针对国内网络环境优化的资源加载器。
    优先检查本地目录，下载失败时提供明确指引，不直接报错崩溃。
    """
    import nltk
    import lemminflect
    
    # 1. 设置本地数据路径 (优先使用当前目录下的 nltk_data 文件夹)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    local_nltk_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(local_nltk_dir, exist_ok=True)
    
    # 强制将本地路径加入 NLTK 搜索路径的首位
    nltk.data.path.insert(0, local_nltk_dir)
    
    # 需要的 NLTK 数据包列表
    required_packages = [
        'averaged_perceptron_tagger', 
        'punkt', 
        'punkt_tab', 
        'wordnet', 
        'omw-1.4'
    ]
    
    missing_packages = []
    
    # 2. 检查包是否存在
    for pkg in required_packages:
        try:
            # 尝试查找 (支持 tokenizers, taggers, corpora 等不同子目录)
            nltk.data.find(f'{pkg}')
        except LookupError:
            # 再试一次具体路径查找，防止 find 没找到但其实在
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError:
                try: nltk.data.find(f'taggers/{pkg}')
                except LookupError:
                    try: nltk.data.find(f'corpora/{pkg}')
                    except LookupError:
                        missing_packages.append(pkg)

    # 3. 尝试下载缺失包 (带异常处理)
    if missing_packages:
        try:
            # 尝试静默下载
            nltk.download(missing_packages, download_dir=local_nltk_dir, quiet=True)
        except Exception as e:
            # 下载失败 (国内常见情况)
            pass

    return nltk, lemminflect, missing_packages

@st.cache_data
def load_vocab_data():
    """
    加载 COCA 词频表。返回 {word: rank} 字典。
    """
    possible_files = ["coca_cleaned.csv", "vocab.csv", "data.csv"]
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            
            w_col = next((c for c in df.columns if 'word' in c), None)
            r_col = next((c for c in df.columns if 'rank' in c), None)
            
            if not w_col or not r_col: return {}

            df = df.dropna(subset=[w_col])
            df[w_col] = df[w_col].astype(str).str.lower().str.strip()
            df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
            
            # 排序去重，保留排名最靠前的
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col], keep='first')
            
            return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        except: return {}
    return {}

# 全局加载
VOCAB_DICT = load_vocab_data()
NLTK_LIB, LEMMA_LIB, MISSING_PKGS = load_nlp_resources()

def get_beijing_time_str():
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime('%m%d_%H%M')

def clear_all_state():
    """完全重置状态"""
    for k in ['gen_words_data', 'raw_count', 'process_time', 'anki_input_text']:
        if k in st.session_state: del st.session_state[k]
    st.session_state['uploader_id'] = str(random.randint(100000, 999999))
    if 'paste_key' in st.session_state: st.session_state['paste_key'] = ""

# ==========================================
# 2. 核心逻辑 (纯 Python 实现，无外部 API 调用)
# ==========================================

def extract_text_from_file(uploaded_file):
    import pypdf, docx, ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    
    text = ""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'txt':
            bytes_data = uploaded_file.getvalue()
            for enc in ['utf-8', 'gb18030', 'gbk', 'latin-1']:
                try: text = bytes_data.decode(enc); break
                except: continue
        elif file_ext == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif file_ext == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_ext == 'epub':
            with open("temp.epub", "wb") as f: f.write(uploaded_file.getvalue())
            book = epub.read_epub("temp.epub")
            parts = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    parts.append(soup.get_text(separator=' ', strip=True))
            text = " ".join(parts)
            if os.path.exists("temp.epub"): os.remove("temp.epub")
    except Exception as e: return f"Error: {e}"
    return text

def is_valid_word(word):
    if len(word) < 2 or len(word) > 25: return False
    if re.search(r'(.)\1{2,}', word): return False # 3个连续相同字母
    if not re.search(r'[aeiouy]', word): return False # 无元音
    if re.search(r'[0-9_]', word): return False
    return True

def analyze_logic(text, min_rank, max_rank, include_unknown):
    # 如果 NLTK 加载失败，提供降级处理
    if MISSING_PKGS:
        return [], 0
        
    raw_tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
    total_words = len(raw_tokens)
    clean_tokens = set([t.lower() for t in raw_tokens if is_valid_word(t.lower())])
    
    final_candidates = []
    seen_lemmas = set()
    
    for w in clean_tokens:
        try: lemma = LEMMA_LIB.getLemma(w, upos='VERB')[0]
        except: lemma = w
            
        rank_lemma = VOCAB_DICT.get(lemma, 99999)
        rank_orig = VOCAB_DICT.get(w, 99999)
        best_rank = min(rank_lemma, rank_orig)
        word_to_keep = lemma if rank_lemma != 99999 else w
        
        if (min_rank <= best_rank <= max_rank) or (include_unknown and best_rank == 99999):
            if lemma not in seen_lemmas:
                final_candidates.append((word_to_keep, best_rank))
                seen_lemmas.add(lemma)
                
    final_candidates.sort(key=lambda x: x[1])
    return final_candidates, total_words

# ==========================================
# 3. Anki 解析与生成 (本地处理)
# ==========================================
def parse_anki_data(raw_text):
    parsed_cards = []
    text = raw_text.replace("```json", "").replace("```", "").strip()
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    seen_phrases = set()

    for match in matches:
        try:
            data = json.loads(match.group(), strict=False)
            front = str(data.get("w", "")).strip().replace('**', '')
            meaning = str(data.get("m", "")).strip()
            if not front or not meaning: continue
            
            if front.lower() in seen_phrases: continue
            seen_phrases.add(front.lower())

            parsed_cards.append({
                'front': front,
                'back': meaning,
                'examples': str(data.get("e", "")).strip(),
                'etymology': str(data.get("r", "")).strip()
            })
        except: continue
    return parsed_cards

def generate_anki_package(cards_data, deck_name):
    import genanki, tempfile
    
    CSS = """
    .card { font-family: arial; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2e2e2e; color: #f0f0f0; }
    .phrase { font-size: 26px; font-weight: bold; color: #0056b3; margin-bottom: 20px; }
    .definition { font-weight: bold; margin-bottom: 15px; font-size: 18px; text-align: left; }
    .examples { background: #f7f9fa; padding: 10px; border-left: 3px solid #0056b3; font-style: italic; font-size: 16px; text-align: left; }
    .nightMode .examples { background: #383838; border-color: #66b0ff; }
    .etymology { font-size: 14px; color: #666; margin-top: 15px; padding-top: 10px; border-top: 1px dashed #ccc; text-align: left; }
    """
    
    model = genanki.Model(
        random.randrange(1<<30, 1<<31), 'VocabFlow Model',
        fields=[{'name': 'Front'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="phrase">{{Front}}</div>',
            'afmt': '{{FrontSide}}<hr><div class="definition">{{Meaning}}</div><div class="examples">{{Examples}}</div><div class="etymology">{{Etymology}}</div>',
        }], css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1<<30, 1<<31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[c['front'], c['back'], c['examples'].replace('\n','<br>'), c['etymology']]))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

def get_ai_prompt(words, front_mode, def_mode, ex_count, need_ety):
    w_list = ", ".join(words)
    w_instr = "Key `w`: The word itself (lemma)." if "单词" in front_mode else "Key `w`: A common short phrase/collocation."
    m_instr = "Key `m`: Concise Chinese definition." if def_mode == "中文" else ("Key `m`: English Definition + Chinese Definition." if def_mode == "中英双语" else "Key `m`: English definition.")
    return f"""Task: Create Anki JSON.\nWords: {w_list}\n\nFormat: NDJSON (One JSON per line).\nKeys: `w` (Front), `m` (Meaning), `e` ({ex_count} Example sentences), `r` ({'Etymology in Chinese' if need_ety else 'Empty string'}).\n\nRequirements:\n1. {w_instr}\n2. {m_instr}\n\nStart:"""

# ==========================================
# 4. 主界面 (UI)
# ==========================================

st.title("⚡️ Vocab Flow Ultra (Stable)")

# ⚠️ NLTK 缺失警告 (针对国内网络)
if MISSING_PKGS:
    st.error(f"""
    **⚠️ 缺少必要的 NLP 数据包 (网络下载失败)**
    
    由于网络原因，NLTK 数据未能自动下载。请手动执行以下操作：
    1. 确保已安装 NLTK: `pip install nltk`
    2. 在 Python 中运行: `import nltk; nltk.download('popular')`
    3. 或者手动下载缺失的包: {', '.join(MISSING_PKGS)}
    """)

if not VOCAB_DICT:
    st.warning("⚠️ 未检测到 `coca_cleaned.csv`，词频筛选功能将失效。请将文件放入根目录。")

tab_guide, tab_extract, tab_anki = st.tabs(["📖 使用指南", "1️⃣ 单词提取", "2️⃣ Anki 制作"])

with tab_guide:
    st.markdown("""
    <div class="guide-step">
    <span class="guide-title">步骤 1: 提取</span>
    上传文件或粘贴文本，系统将自动进行词形还原并按词频筛选。
    </div>
    <div class="guide-step">
    <span class="guide-title">步骤 2: 生成 Prompt</span>
    复制生成的 Prompt 发送给 AI。
    </div>
    <div class="guide-step">
    <span class="guide-title">步骤 3: 制作 Anki</span>
    粘贴 AI 回复的 JSON，生成 <code>.apkg</code> 导入包。
    </div>
    """, unsafe_allow_html=True)

with tab_extract:
    c1, c2 = st.columns(2)
    # 按要求设置: 默认8000/15000, 步长500
    min_r = c1.number_input("忽略排名前 N (太简单的词)", 1, 20000, 8000, step=500)
    max_r = c2.number_input("忽略排名后 N (太生僻的词)", 1000, 50000, 15000, step=500)
    include_unknown = st.checkbox("🔓 包含无排名词汇 (人名/新词)", value=False)
    
    uploaded_file = st.file_uploader("📂 上传文件 (支持 PDF/DOCX/EPUB/TXT)", key=st.session_state['uploader_id'])
    pasted_text = st.text_area("📄 ...或在此粘贴文本", height=100, key="paste_key")
    
    col_b1, col_b2 = st.columns([1, 4])
    with col_b1: st.button("🗑️ 清空", on_click=clear_all_state)
    with col_b2: run_btn = st.button("🚀 开始分析", type="primary", disabled=bool(MISSING_PKGS))

    if run_btn and not MISSING_PKGS:
        txt = extract_text_from_file(uploaded_file) if uploaded_file else pasted_text
        if len(txt.strip()) > 5:
            with st.spinner("正在分析..."):
                t0 = time.time()
                data, raw_c = analyze_logic(txt, min_r, max_r, include_unknown)
                st.session_state['gen_words_data'] = data
                st.session_state['raw_count'] = raw_c
                st.session_state['process_time'] = time.time() - t0
        else:
            st.warning("⚠️ 内容太短或无效")

    if st.session_state.get('gen_words_data'):
        data = st.session_state['gen_words_data']
        words = [x[0] for x in data]
        
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("原文词数", f"{st.session_state['raw_count']:,}")
        m2.metric("提取生词", f"{len(words)}")
        m3.metric("耗时", f"{st.session_state['process_time']:.2f}s")
        
        with st.expander("📋 生词列表预览", expanded=False):
            show_rank = st.checkbox("显示排名")
            disp = ", ".join([f"{w}({r})" if show_rank else w for w, r in data])
            st.markdown(f'<div class="scrollable-text">{disp}</div>', unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Prompt 设置")
        pc1, pc2, pc3 = st.columns(3)
        fm = pc1.selectbox("正面", ["单词 (Word)", "短语 (Phrase)"])
        dm = pc2.selectbox("释义", ["英文", "中文", "中英双语"])
        # 按要求设置: 默认100, 最大150, 最小1, 步长1
        bs = pc3.number_input("每组数量", min_value=1, max_value=150, value=100, step=1)
        
        batches = [words[i:i+bs] for i in range(0, len(words), bs)]
        st.info(f"共生成 {len(batches)} 组 Prompts")
        
        for i, batch in enumerate(batches):
            with st.expander(f"📝 第 {i+1} 组 ({len(batch)} 词)"):
                st.code(get_ai_prompt(batch, fm, dm, 1, True), language="text")

with tab_anki:
    st.caption("👇 将 AI 返回的 JSON 粘贴到此处 (支持多次追加):")
    ai_in = st.text_area("JSON 输入", height=200, key="anki_input_text")
    d_name = st.text_input("牌组名", f"Vocab_{get_beijing_time_str()}")
    
    if st.button("🛠️ 生成 .apkg", type="primary"):
        if ai_in.strip():
            cards = parse_anki_data(ai_in)
            if cards:
                st.success(f"成功解析 {len(cards)} 张卡片")
                st.dataframe(pd.DataFrame(cards)[['front','back','etymology']], use_container_width=True)
                apk = generate_anki_package(cards, d_name)
                with open(apk, "rb") as f:
                    st.download_button(f"📥 下载 {d_name}.apkg", f, file_name=f"{d_name}.apkg")
            else: st.error("未找到有效 JSON 数据")
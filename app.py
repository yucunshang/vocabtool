import streamlit as st
import pandas as pd
import re
import os
import json
import time
import requests
import concurrent.futures
import lemminflect
import nltk
import tempfile
import random
from pathlib import Path

# ==========================================
# 0. 依赖检查与 NLTK 初始化
# ==========================================
try:
    import PyPDF2
    import docx
    import genanki
except ImportError:
    st.error("⚠️ 缺少必要依赖。请在终端运行: pip install PyPDF2 python-docx genanki")
    st.stop()

def download_nltk_resources():
    """静默下载必要的 NLTK 数据"""
    resources = ['punkt', 'averaged_perceptron_tagger', 'names', 'wordnet', 'omw-1.4']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}') if r == 'punkt' else nltk.data.find(f'corpora/{r}')
        except LookupError:
            nltk.download(r, quiet=True)

download_nltk_resources()

# ==========================================
# 1. 页面配置 & 样式优化 (无侧边栏)
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* 隐藏顶部 Hamburger 菜单和默认 Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 强制隐藏侧边栏 (防止误触) */
    [data-testid="stSidebar"] { display: none; }
    
    /* 字体与排版优化 */
    .stCode { font-family: 'Consolas', 'Courier New', monospace !important; }
    .block-container { padding-top: 2rem; padding-bottom: 5rem; max-width: 1200px; }
    
    /* 移动端适配：调整指标数字大小 */
    @media (max-width: 640px) {
        [data-testid="stMetricValue"] { font-size: 22px !important; }
        .stButton button { width: 100%; border-radius: 8px; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. API Key 获取 (使用 st.secrets)
# ==========================================
def get_api_key():
    """优先从 st.secrets 获取，安全且隐蔽"""
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None

API_KEY = get_api_key()

# ==========================================
# 3. 数据加载 (JSON 支持)
# ==========================================
@st.cache_data
def load_data_resources():
    data_dir = Path("data")
    
    # 1. Safe Names (基础 + 扩展)
    safe_names = set(nltk.corpus.names.words())
    if (data_dir / "safe_names.json").exists():
        with open(data_dir / "safe_names.json", "r", encoding="utf-8") as f:
            safe_names.update(json.load(f))
            
    # 2. Terms & Proper Nouns
    tech_terms = json.load(open(data_dir / "terms.json", encoding="utf-8")) if (data_dir / "terms.json").exists() else {}
    proper_map = json.load(open(data_dir / "proper.json", encoding="utf-8")) if (data_dir / "proper.json").exists() else {}
    
    # 3. Global Ranks (COCA)
    entity_ranks = {}
    if (data_dir / "global_ranks.json").exists():
        with open(data_dir / "global_ranks.json", "r", encoding="utf-8") as f:
            entity_ranks = json.load(f)
    else:
        # Fallback 防止报错
        entity_ranks = {"the": 1, "be": 2, "python": 500, "code": 600} 
            
    return safe_names, tech_terms, proper_map, entity_ranks

SAFE_NAMES, TECH_TERMS, PROPER_MAP, GLOBAL_RANKS = load_data_resources()

# ==========================================
# 4. 核心 NLP 处理逻辑
# ==========================================
def clean_text(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_word_info(word):
    w_lower = word.lower()
    # 术语库优先
    if w_lower in TECH_TERMS: return w_lower, 0, 'Tech', TECH_TERMS[w_lower]
    
    # 词形还原
    lemma = lemminflect.getLemma(w_lower, upos="VERB")[0]
    if lemma not in GLOBAL_RANKS:
        lemma = lemminflect.getLemma(w_lower, upos="NOUN")[0]
        
    rank = GLOBAL_RANKS.get(lemma, 99999)
    display = PROPER_MAP.get(lemma, lemma) # 修正大小写
    return display, rank, 'General', ''

def process_text(text, min_rank, top_n):
    tokens = nltk.word_tokenize(text)
    # 过滤逻辑：纯字母，长度>2，非人名
    valid_words = [w for w in tokens if w.isalpha() and len(w)>2 and w not in SAFE_NAMES]
    
    data = []
    seen = set()
    for w in valid_words:
        lemma, rank, cat, tag = get_word_info(w)
        if lemma not in seen:
            seen.add(lemma)
            data.append({"raw": w, "lemma": lemma, "rank": rank, "category": cat})
            
    df = pd.DataFrame(data)
    if df.empty: return df
    
    # 筛选与排序
    df = df[df['rank'] > min_rank].sort_values('rank')
    
    # 标记 Top N
    df['is_top'] = False
    if not df.empty:
        df.iloc[:top_n, df.columns.get_loc('is_top')] = True
        
    return df

# ==========================================
# 5. AI & Anki 生成逻辑
# ==========================================
def get_prompt(lang):
    """V2 Prompt: 包含 One-Shot Example"""
    ex_out = '"book","<b>book</b> [n.]<br>书，书籍<br><em>I read a book.</em>"' if lang == "Chinese" else '"book","<b>book</b> [n.]<br>A written work...<br><em>I read a book.</em>"'
    return f"""Role: Expert Linguist.
Task: Create Anki cards.
Format: CSV "Front","Back"
Rules:
1. Front: The word.
2. Back: Definition in {lang} + 1 Example (wrapped in <em>).
3. Output ONLY CSV lines.

Example:
Input: book
Output:
{ex_out}

Words:
"""

def generate_anki_pkg(cards, deck_name="VocabMaster"):
    """生成 .apkg 文件"""
    model_id = random.randrange(1 << 30, 1 << 31)
    deck_id = random.randrange(1 << 30, 1 << 31)
    
    my_model = genanki.Model(
        model_id, 'VocabMaster Model',
        fields=[{'name': 'Front'}, {'name': 'Back'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div style="text-align:center; font-size:28px; font-weight:bold; color:#333;">{{Front}}</div>',
            'afmt': '{{FrontSide}}<hr id="answer"><div style="text-align:left; font-size:18px; line-height:1.6;">{{Back}}</div>',
        }]
    )
    my_deck = genanki.Deck(deck_id, deck_name)
    for f, b in cards:
        my_deck.add_note(genanki.Note(model=my_model, fields=[f, b]))
        
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.apkg')
    genanki.Package(my_deck).write_to_file(tmp.name)
    return tmp.name

def call_ai_batch(prompt):
    """调用 AI 接口"""
    if not API_KEY: return "Error: API Key missing"
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat", # 根据实际情况修改模型名称
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        r = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=60)
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 6. UI 主程序
# ==========================================
st.title("🚀 Vocab Master Pro")

# --- 参数配置区 (折叠面板) ---
with st.expander("⚙️ 筛选参数设置 (Settings)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1: min_rank = st.number_input("忽略前 N 高频词", 0, 20000, 3000, 500)
    with c2: top_n = st.number_input("精选 Top N", 10, 500, 50, 10)
    with c3: 
        if not API_KEY:
            st.error("❌ 未检测到 Secrets Key")
        else:
            st.success("✅ API Key 已连接")

# --- 输入区 ---
raw_text = ""
tab1, tab2 = st.tabs(["📝 文本输入", "📂 文件上传"])
with tab1:
    txt_in = st.text_area("在此粘贴...", height=150)
    if txt_in: raw_text = txt_in
with tab2:
    up_file = st.file_uploader("支持 PDF, Docx, Txt", type=['pdf', 'docx', 'txt'])
    if up_file:
        if up_file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(up_file)
            raw_text = " ".join([p.extract_text() for p in reader.pages])
        elif up_file.name.endswith(".docx"):
            doc = docx.Document(up_file)
            raw_text = " ".join([p.text for p in doc.paragraphs])
        else:
            raw_text = up_file.read().decode("utf-8")

# --- 分析按钮 ---
if st.button("🚀 开始分析 (Analyze)", type="primary", use_container_width=True):
    if not raw_text:
        st.warning("请先输入内容")
    else:
        with st.spinner("NLP 处理中..."):
            st.session_state.df = process_text(clean_text(raw_text), min_rank, top_n)

# --- 结果区 ---
if "df" in st.session_state and not st.session_state.df.empty:
    df = st.session_state.df
    top_df = df[df['is_top']].copy()
    
    st.divider()
    # 指标展示
    m1, m2, m3 = st.columns(3)
    m1.metric("总词汇", len(df))
    m2.metric("重点词", len(top_df))
    m3.metric("难度系数", int(df['rank'].mean()) if not df.empty else 0)
    
    t_res1, t_res2 = st.tabs(["🔥 重点词 & 制卡", "📋 完整列表"])
    
    with t_res1:
        st.dataframe(top_df[['lemma', 'rank', 'category']], use_container_width=True)
        
        st.markdown("### 🤖 AI 制卡 (Anki)")
        if not API_KEY:
            st.warning("⚠️ 请先在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY 才能使用 AI 功能")
        else:
            col_a, col_b = st.columns(2)
            with col_a: lang = st.selectbox("释义语言", ["Chinese", "English"])
            with col_b: fmt = st.selectbox("导出格式", ["Anki (.apkg)", "CSV"])
            
            # 预览按钮
            if st.button("👁️ 预览首词 (Preview 1 Card)"):
                word = top_df.iloc[0]['lemma']
                p = get_prompt(lang) + word
                with st.spinner("生成预览..."):
                    st.code(call_ai_batch(p), language="csv")
            
            # 批量生成按钮
            if st.button("⚡ 生成全部卡片 (Batch Generate)", type="primary"):
                words = top_df['lemma'].tolist()
                batches = [words[i:i+10] for i in range(0, len(words), 10)]
                all_cards = []
                
                prog_bar = st.progress(0)
                status_txt = st.empty()
                
                # 并发处理
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {}
                    for i, batch in enumerate(batches):
                        p = get_prompt(lang) + "\n".join(batch)
                        futures[executor.submit(call_ai_batch, p)] = batch
                    
                    done_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        # 简单的 CSV 解析
                        for line in res.strip().split('\n'):
                            parts = line.split('","')
                            if len(parts) >= 2:
                                all_cards.append((parts[0].strip('"'), parts[-1].strip('"')))
                        
                        done_count += 1
                        prog_bar.progress(done_count / len(batches))
                        status_txt.text(f"已处理: {done_count}/{len(batches)} 批次")
                
                st.success(f"生成完成！共 {len(all_cards)} 张卡片")
                
                # 下载逻辑
                if fmt == "Anki (.apkg)":
                    path = generate_anki_pkg(all_cards)
                    with open(path, "rb") as f:
                        st.download_button("📥 下载 Anki 牌组 (.apkg)", f, file_name="vocab.apkg", mime="application/apkg")
                else:
                    csv_data = "\n".join([f'"{f}","{b}"' for f,b in all_cards])
                    st.download_button("📥 下载 CSV", csv_data, file_name="vocab.csv")

    with t_res2:
        st.dataframe(df, use_container_width=True)
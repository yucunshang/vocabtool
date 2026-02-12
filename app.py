import streamlit as st
import pandas as pd
import re
import os
import io
import time
import lemminflect
import nltk
import genanki
import random
import tempfile
from bs4 import BeautifulSoup

# --- 文件处理库 ---
import pypdf
import docx
import ebooklib
from ebooklib import epub

# ==========================================
# 0. 页面基础配置 & 样式
# ==========================================
st.set_page_config(
    page_title="Vocab Flow Ultra", 
    page_icon="⚡️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 14px; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; }
    .stat-box { padding: 10px; background-color: #f0f2f6; border-radius: 8px; margin-bottom: 10px; text-align: center; }
    .copy-hint { font-size: 0.8em; color: #888; margin-top: -10px; margin-bottom: 10px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 资源加载 & 工具函数
# ==========================================
@st.cache_resource
def setup_nltk():
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(root_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        for pkg in ['averaged_perceptron_tagger', 'punkt', 'punkt_tab']:
            try: nltk.data.find(f'tokenizers/{pkg}')
            except LookupError: nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
    except: pass
setup_nltk()

@st.cache_data
def load_vocab_data():
    """加载词频表"""
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

def get_lemma(word):
    try: return lemminflect.getLemma(word, upos='VERB')[0]
    except: return word

# ==========================================
# 2. 多格式文件解析 (百万字优化版)
# ==========================================
def extract_text_from_file(uploaded_file):
    """根据文件类型提取文本"""
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            
        elif file_type == 'pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
            
        elif file_type == 'epub':
            # 需要先保存为临时文件才能用 ebooklib 读取
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            book = epub.read_epub(tmp_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + " "
            os.remove(tmp_path)
            
    except Exception as e:
        return f"Error: {e}"
        
    return text

@st.cache_data
def fast_analyze_text(text, current_lvl, target_lvl):
    """
    性能优化核心：
    1. 使用正则 (re) 替代 NLTK 做分词，速度快 10 倍。
    2. 使用 Set 去重后才做 Lemmatization，避免对 100 万个词重复运算。
    """
    # 1. 快速分词 (Regex)
    raw_tokens = re.findall(r"[a-z]+", text.lower())
    
    # 2. 统计总词数
    total_words = len(raw_tokens)
    
    # 3. 核心算法：仅对去重后的词进行词形还原和查表
    unique_tokens = set(raw_tokens)
    target_words = []
    
    for w in unique_tokens:
        if len(w) < 3: continue # 忽略过短单词
        lemma = get_lemma(w)
        rank = VOCAB_DICT.get(lemma, 99999)
        
        # 筛选逻辑
        if rank > current_lvl and rank <= target_lvl:
            target_words.append((lemma, rank))
            
    # 4. 排序
    target_words.sort(key=lambda x: x[1])
    final_list = [x[0] for x in target_words]
    
    return final_list, total_words

# ==========================================
# 3. Anki 打包 & CSS (保持不变)
# ==========================================
def generate_anki_package(cards_data, deck_name="Vocab_Deck"):
    CSS = """
    .card { font-family: arial; font-size: 20px; text-align: center; color: #333; background-color: white; padding: 20px; }
    .nightMode .card { background-color: #2f2f31; color: #f5f5f5; }
    .word { font-size: 40px; font-weight: bold; color: #007AFF; margin-bottom: 10px; }
    .nightMode .word { color: #5FA9FF; }
    .phonetic { color: #888; font-size: 18px; font-family: sans-serif; }
    .def-container { text-align: left; margin-top: 20px; border-top: 1px solid #ddd; padding-top: 10px; }
    .definition { font-weight: bold; color: #444; margin-bottom: 10px; }
    .nightMode .definition { color: #ddd; }
    .examples { background: #f4f4f4; padding: 10px; border-radius: 5px; color: #555; font-style: italic; font-size: 16px; }
    .nightMode .examples { background: #333; color: #ccc; }
    .etymology { margin-top: 15px; font-size: 14px; color: #888; border: 1px dashed #ccc; padding: 5px; display: inline-block;}
    """
    
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31), 'VocabFlow Model',
        fields=[{'name': 'Word'}, {'name': 'IPA'}, {'name': 'Meaning'}, {'name': 'Examples'}, {'name': 'Etymology'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '<div class="word">{{Word}}</div><div class="phonetic">{{IPA}}</div>',
            'afmt': '{{FrontSide}}<div class="def-container"><div class="definition">{{Meaning}}</div><div class="examples">{{Examples}}</div><div class="etymology">🌱 {{Etymology}}</div></div>',
        }], css=CSS
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_name)
    for c in cards_data:
        deck.add_note(genanki.Note(model=model, fields=[c['word'], c['ipa'], c['meaning'], c['examples'].replace('\n','<br>'), c['etymology']]))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.apkg') as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        return tmp.name

def get_ai_prompt(words):
    w_list = ", ".join(words)
    return f"""
Act as a Lexicographer. Create Anki card data.
Words: {w_list}

**Strict Output Format (Pipe Separated `|`, NO Header):**
Word | IPA | Concise English Definition | 2 English Sentences | Etymology (Root+Suffix)

**Requirements:**
1. Definition: Simple English (B2 level).
2. Examples: 2 sentences separated by `<br>`.
3. Etymology: Format `root(meaning) + suffix`.
4. NO Header row.

**Example:**
benevolent | /bəˈnevələnt/ | kind and meaningful | He is benevolent.<br>A benevolent fund. | bene(good) + vol(wish)
"""

# ==========================================
# 4. 主界面
# ==========================================
st.title("⚡️ Vocab Flow Ultra")

if not VOCAB_DICT:
    st.error("⚠️ 缺失 `coca_cleaned.csv`，无法进行频率筛选！")

tab1, tab2, tab3 = st.tabs(["📂 文件分析", "🔢 词频生成", "🛠️ 制作 Anki"])

# --- Tab 1: 文件分析 (支持百万字) ---
with tab1:
    c1, c2 = st.columns(2)
    curr = c1.number_input("忽略简单词 (Rank <)", 1000, 20000, 4000, step=500)
    targ = c2.number_input("忽略生僻词 (Rank >)", 2000, 50000, 15000, step=500)
    
    # 支持多种格式上传
    uploaded_file = st.file_uploader("上传文件 (支持 .txt, .pdf, .docx, .epub)", type=['txt', 'pdf', 'docx', 'epub'])
    
    if uploaded_file and st.button("🚀 开始极速分析"):
        with st.spinner("正在解析文件..."):
            # 1. 解析文本
            raw_text = extract_text_from_file(uploaded_file)
            
            if len(raw_text) < 10:
                st.error("无法读取文本，可能是扫描版PDF或加密文件。")
            else:
                # 2. 极速分析
                t0 = time.time()
                final_words, total_count = fast_analyze_text(raw_text, curr, targ)
                t1 = time.time()
                
                st.markdown(f"""
                <div class="stat-box">
                    📊 原文约 <b>{total_count}</b> 词 | 耗时 <b>{t1-t0:.2f}s</b><br>
                    🎯 筛选出 <b>{len(final_words)}</b> 个重点词 (Rank {curr}-{targ})
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state['gen_words'] = final_words

# --- Tab 2: 词频生成 (保留的功能) ---
with tab2:
    st.caption("直接根据词频排名生成单词表，无需上传文件。")
    c_a, c_b = st.columns(2)
    start_rank = c_a.number_input("起始排名 (Start Rank)", 1, 20000, 8000, step=100)
    count_num = c_b.number_input("生成数量 (Count)", 10, 500, 50, step=10)
    
    if st.button("🔢 生成列表", type="primary"):
        if FULL_DF is not None:
            # 这里的 FULL_DF 是在 load_data 里返回的原始 DataFrame
            # 我们需要 FULL_DF 的 columns 分别是 word 和 rank
            # 在 load_vocab_data 稍微调整一下让它返回 DF
             
            # 筛选逻辑
            try:
                # 找到 Rank 列名
                r_col = next(c for c in FULL_DF.columns if 'rank' in c)
                w_col = next(c for c in FULL_DF.columns if 'word' in c)
                
                subset = FULL_DF[FULL_DF[r_col] >= start_rank].sort_values(r_col).head(count_num)
                gen_list = subset[w_col].tolist()
                st.session_state['gen_words'] = gen_list
                st.success(f"已生成 {len(gen_list)} 个单词 (Rank {start_rank} 起)")
            except Exception as e:
                st.error(f"生成失败: {e}")
        else:
            st.error("无数据源")

# --- 结果展示与 Prompt 生成 (Tab 1 & 2 共用) ---
if 'gen_words' in st.session_state and st.session_state['gen_words']:
    st.divider()
    st.markdown("### 📋 单词列表 & Prompt")
    
    words = st.session_state['gen_words']
    words_str = ", ".join(words)
    
    # 1. 提供一键复制的 Code Block
    st.markdown("<div class='copy-hint'>👇 点击代码块右上角即可一键复制单词表</div>", unsafe_allow_html=True)
    st.code(words_str, language="text")
    
    # 2. 生成 AI Prompt
    if st.button("🤖 生成 AI Prompt"):
        prompt = get_ai_prompt(words)
        st.code(prompt, language="markdown")
        st.info("复制上方 Prompt 发送给 AI，然后将结果粘贴到 '制作 Anki' 页面。")

# --- Tab 3: 制作 Anki ---
with tab3:
    st.markdown("### 🛠️ 制作 iOS 适配包 (.apkg)")
    ai_resp = st.text_area("粘贴 AI 回复 (Word | IPA | Def | Ex | Etym)", height=200)
    deck_name = st.text_input("牌组名", "My Deck")
    
    if st.button("📦 打包下载"):
        if not ai_resp.strip(): st.error("内容为空")
        else:
            cards = []
            for line in ai_resp.strip().split('\n'):
                if "|" not in line or "Word |" in line: continue
                p = [x.strip() for x in line.split('|')]
                if len(p) >= 3:
                    cards.append({'word':p[0], 'ipa':p[1] if len(p)>1 else '', 'meaning':p[2] if len(p)>2 else '', 'examples':p[3] if len(p)>3 else '', 'etymology':p[4] if len(p)>4 else ''})
            
            if cards:
                f_path = generate_anki_package(cards, deck_name)
                with open(f_path, "rb") as f:
                    st.download_button(f"📥 下载 {deck_name}.apkg", f, file_name=f"{deck_name}.apkg", mime="application/octet-stream", type="primary")
                st.success(f"成功打包 {len(cards)} 张卡片！")
            else:
                st.error("无有效数据，请检查分隔符 |")
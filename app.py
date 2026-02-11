import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk
import json
import zipfile

# 尝试导入文档处理库
try:
    import PyPDF2
    import docx
except ImportError:
    pass # 手机端如果只是刷频段，不需要这些，容错处理

# ==========================================
# 1. 移动端优先配置
# ==========================================
st.set_page_config(page_title="Vocab Prompt Gen", page_icon="📱", layout="centered", initial_sidebar_state="collapsed")

# CSS 适配手机端：增大间距，隐藏不需要的元素
st.markdown("""
<style>
    /* 全局字体与间距优化 */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* 隐藏 Streamlit 默认汉堡菜单和页脚，不仅清爽也防误触 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 按钮样式优化 - 更像 App 的触控区 */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        font-weight: bold;
    }
    
    /* 输入框样式 */
    .stTextArea>div>div>textarea {
        font-size: 16px; /* 防止 iOS 输入缩放 */
    }
    
    /* 统计数据大字号 */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
    }
    
    /* 分割线颜色 */
    hr { margin: 1.5em 0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据初始化 (精简版)
# ==========================================
@st.cache_data
def load_resources():
    # 路径检查
    if not os.path.exists('data'): return {}, {}, {}
    
    try:
        with open('data/terms.json', 'r', encoding='utf-8') as f: terms = json.load(f)
        with open('data/proper.json', 'r', encoding='utf-8') as f: proper = json.load(f)
        # 加载词频表
        vocab = {}
        file_path = next((f for f in ["coca_cleaned.csv", "data.csv"] if os.path.exists(f)), None)
        if file_path:
            df = pd.read_csv(file_path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            w_col = next((c for c in cols if 'word' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c), cols[1])
            # 简单清洗
            df = df.sort_values(r_col).drop_duplicates(subset=[w_col])
            vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
            
        return terms, proper, vocab
    except Exception as e:
        return {}, {}, {}

BUILTIN_TERMS, PROPER_NOUNS, VOCAB_DICT = load_resources()

# 词形还原需 NLTK
@st.cache_resource
def setup_nltk():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(root_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    try: 
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    except: pass
setup_nltk()

def get_lemma(w):
    lemmas = lemminflect.getAllLemmas(w)
    if not lemmas: return w.lower()
    return list(lemmas.values())[0][0]

# ==========================================
# 3. 核心功能函数
# ==========================================

# 提取文本
def extract_text(file_obj):
    if not file_obj: return ""
    ext = file_obj.name.split('.')[-1].lower()
    try:
        if ext == 'txt': return file_obj.getvalue().decode("utf-8", errors="ignore")
        if ext == 'pdf':
            reader = PyPDF2.PdfReader(file_obj)
            return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
        if ext == 'docx':
            doc = docx.Document(file_obj)
            return " ".join([p.text for p in doc.paragraphs])
        if ext == 'epub':
            # 简化处理 epub
            with zipfile.ZipFile(file_obj) as z:
                return " ".join([z.read(n).decode('utf-8', errors='ignore') for n in z.namelist() if n.endswith('.html')])
    except: return ""
    return ""

# 生成 Prompt
def generate_prompt(words, start_rank, end_rank, source_type="rank"):
    word_list_str = ", ".join(words)
    count = len(words)
    
    # 针对手机端优化的 Prompt
    prompt = f"""You are an expert Anki card generator.
    
TASK:
Create vocabulary flashcards for the following {count} English words.
{'Source: Words ranked ' + str(start_rank) + '-' + str(end_rank) + ' in frequency.' if source_type == 'rank' else 'Source: Extracted from user text.'}

STRICT OUTPUT FORMAT:
Please generate a **downloadable CSV file** with the following columns. 
If you cannot generate a file, output a **Code Block** in CSV format that I can easily copy.

CSV Structure:
"Target Word (w/ POS)","Definition & Context"

Content Rules:
1. Column 1: The word + Part of Speech (e.g., "ephemeral (adj)").
2. Column 2: 
   - English Definition (brief & clear).
   - Chinese Definition (brief).
   - One high-quality example sentence with the target word **bolded** (use HTML <b>word</b>).
   - [Optional] Etymology/Root if helpful.
   - Format usage: Use HTML line breaks <br> to separate definition and example.

List of Words:
{word_list_str}
"""
    return prompt

# ==========================================
# 4. 界面逻辑 (App UI)
# ==========================================

st.title("📱 Vocab Master")

# 模式切换：如同 App 的底部或顶部 Tab
mode = st.radio("功能模式", ["🔢 词频刷词 (Rank)", "📖 文本透视 (Context)"], horizontal=True, label_visibility="collapsed")

# -------------------------------------------------
# 模式 A: 词频刷词 (Range Mode) - 新功能
# -------------------------------------------------
if "Rank" in mode:
    st.markdown("### 🎯 制定每日刷词计划")
    
    # 将字典反转用于查找：Rank -> List of Words
    # 注意：可能有多个词拥有相同的 Rank，虽然我们的清洗逻辑尽量避免了
    if 'rank_map' not in st.session_state:
        r_map = {}
        for w, r in VOCAB_DICT.items():
            if r not in r_map: r_map[r] = []
            r_map[r].append(w)
        st.session_state.rank_map = r_map

    col1, col2 = st.columns(2)
    with col1:
        start_r = st.number_input("起始排名", value=8000, step=100)
    with col2:
        end_r = st.number_input("结束排名", value=8100, step=100)
        
    if start_r >= end_r:
        st.error("结束排名必须大于起始排名")
    else:
        # 获取该区间的词
        target_words = []
        for r in range(start_r, end_r + 1):
            if r in st.session_state.rank_map:
                target_words.extend(st.session_state.rank_map[r])
        
        # 截断一下防止过多
        if len(target_words) > 100:
            st.warning(f"区间内有 {len(target_words)} 个词，自动截取前 100 个。")
            target_words = target_words[:100]
            
        st.info(f"✅ 选中 **{len(target_words)}** 个单词")
        
        with st.expander("👀 预览单词列表"):
            st.write(", ".join(target_words))
            
        if st.button("🚀 生成 AI Prompt", type="primary"):
            final_prompt = generate_prompt(target_words, start_r, end_r, "rank")
            st.session_state.final_prompt = final_prompt

# -------------------------------------------------
# 模式 B: 文本透视 (Context Mode) - 原功能简化
# -------------------------------------------------
else:
    st.markdown("### 📖 从阅读材料中提取")
    
    # 隐藏的高级设置
    with st.expander("⚙️ 过滤设置 (默认已优化)"):
        user_level = st.slider("忽略过于简单的词 (Rank < X)", 0, 15000, 4000)
        max_level = st.slider("忽略过于生僻的词 (Rank > X)", 1000, 30000, 20000)
    
    # 输入区：手机上 Text Area 不好用，优先文件，或者粘贴板
    tab1, tab2 = st.tabs(["📝 粘贴文本", "📂 上传文档"])
    with tab1:
        text_input = st.text_area("在此粘贴", height=150, placeholder="支持长按粘贴...")
    with tab2:
        file_input = st.file_uploader("支持 TXT/PDF/DOCX", type=["txt", "pdf", "docx", "epub"])
    
    if st.button("🔍 分析并提取生词", type="primary"):
        # 处理文本
        raw_text = text_input
        if file_input: raw_text += "\n" + extract_text(file_input)
        
        if not raw_text.strip():
            st.warning("没内容啊大佬")
        else:
            # 简单的 NLP 处理
            words = re.findall(r"[a-zA-Z']+", raw_text)
            lemmas = set([get_lemma(w).lower() for w in words])
            
            # 过滤逻辑
            valid_words = []
            for w in lemmas:
                rank = VOCAB_DICT.get(w, 99999)
                if user_level < rank <= max_level:
                    valid_words.append((w, rank))
            
            # 排序
            valid_words.sort(key=lambda x: x[1])
            final_list = [x[0] for x in valid_words]
            
            # 截取 Top 50 (手机上不宜一次太多)
            if len(final_list) > 50:
                final_list = final_list[:50]
                st.caption("📱 为方便手机制卡，仅保留 Top 50 生词")
                
            st.success(f"筛选出 {len(final_list)} 个生词")
            with st.expander("👀 预览单词"):
                st.write(", ".join(final_list))
                
            st.session_state.final_prompt = generate_prompt(final_list, 0, 0, "text")

# ==========================================
# 5. 结果输出区 (共用)
# ==========================================
if "final_prompt" in st.session_state:
    st.divider()
    st.markdown("### 📋 复制指令给 AI")
    st.info("💡 这是一个针对移动端优化的指令。复制后，发送给 ChatGPT/Claude App 即可。")
    
    # 使用代码块显示，右上角自带复制按钮
    st.code(st.session_state.final_prompt, language="markdown")
    
    st.markdown("""
    **手机端使用技巧：**
    1. 点击上方代码块右上角的 **Copy**。
    2. 打开 ChatGPT App 粘贴发送。
    3. AI 生成后，点击下载 CSV 文件。
    4. 用 **AnkiMobile** 打开该文件即可直接导入。
    """)
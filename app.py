import streamlit as st
import pandas as pd
import re
import os
import sys
import subprocess

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🧠", layout="wide")

# ==========================================
# 0. 核心：自动环境修复 (Self-Healing)
# ==========================================
@st.cache_resource
def load_spacy_model():
    """
    顽强的加载器：
    1. 检查有没有 spacy，没有就报错
    2. 检查有没有模型 en_core_web_sm，没有就现场下载
    """
    try:
        import spacy
    except ImportError:
        st.error("❌ 严重错误：requirements.txt 未生效，找不到 spacy 库。请尝试 Reboot App。")
        st.stop()

    model_name = "en_core_web_sm"
    try:
        # 尝试直接加载
        nlp = spacy.load(model_name)
    except OSError:
        # 如果报错说找不到模型，就调用命令行下载
        st.warning(f"正在自动下载语言模型 {model_name}... (初次运行需要 1 分钟)")
        try:
            # 使用 subprocess 调用安装命令
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            nlp = spacy.load(model_name)
        except Exception as e:
            st.error(f"模型下载失败，请检查网络或日志: {e}")
            st.stop()
            
    return nlp

# 加载 NLP 引擎
nlp = load_spacy_model()
st.sidebar.success("✅ spaCy 引擎已就绪")

# ==========================================
# 1. 词库加载 (针对 coca_cleaned.csv 优化)
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]

@st.cache_data
def load_vocab():
    file_path = None
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            file_path = f
            break
            
    if not file_path: return None, "未找到文件"

    # 优先读 coca_cleaned (标准格式)
    if 'cleaned' in file_path:
        try:
            # 你的 coca_cleaned.csv 是标准的 word, rank 格式
            df = pd.read_csv(file_path)
            # 强制小写去空格
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            # 建立索引
            vocab = pd.Series(df['rank'].values, index=df['word']).to_dict()
            return vocab, "加载成功 (Cleaned)"
        except: pass

    # 兜底读原始文件
    for enc in ['utf-8', 'utf-8-sig', 'gbk']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            cols = [str(c).lower() for c in df.columns]
            df.columns = cols
            
            w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
            r_col = next((c for c in cols if 'rank' in c or '排序' in c or '词频' in c), cols[1] if len(cols)>1 else cols[0])
            
            df['w'] = df[w_col].astype(str).str.lower().str.strip()
            df['r'] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
            
            vocab = pd.Series(df['r'].values, index=df['w']).to_dict()
            return vocab, "加载成功 (Raw)"
        except: continue
        
    return None, "加载失败"

vocab_dict, msg = load_vocab()

# ==========================================
# 2. 界面显示
# ==========================================
st.title("🧠 Vibe Vocab v11.0 (自动修复版)")
st.caption("spaCy 驱动 · 自动下载模型 · 彻底解决还原问题")

if not vocab_dict:
    st.error(msg)
    st.stop()

st.sidebar.info(f"📚 词库: {msg}")

# === 验证区 ===
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 还原 & 排名测试", expanded=True):
    # 测试还原
    doc = nlp("families are better")
    res = [token.lemma_ for token in doc]
    st.write(f"families are better -> {res}")
    
    # 测试排名
    check_be = vocab_dict.get('be', 'Not Found')
    st.write(f"'be' rank: {check_be}")
    
    if check_be == 'Not Found' or check_be > 100:
        st.error("⚠️ 词库读取可能有误，'be' 的排名不对！")
    else:
        st.success("✅ 词库读取正常")

vocab_range = st.sidebar.slider("学习区间", 1, 20000, (6000, 8000), 500)
r_start, r_end = vocab_range

# ==========================================
# 3. 核心处理逻辑 (spaCy)
# ==========================================
def process_text(text):
    # 使用 spaCy 处理整个文本
    doc = nlp(text.lower())
    
    seen_lemmas = set()
    unique_items = []
    
    for token in doc:
        # 只保留字母，且长度大于1
        if token.is_alpha and len(token.text) > 1:
            lemma = token.lemma_.lower() # 这里拿到的就是 family, be, go
            original = token.text.lower()
            
            if lemma not in seen_lemmas:
                unique_items.append((original, lemma))
                seen_lemmas.add(lemma)
    
    # 按排名排序逻辑
    # 我们先查 rank，再排序
    processed_list = []
    
    for original, lemma in unique_items:
        rank = 99999
        match = lemma # 默认用还原后的词(family)去查
        note = ""

        # 1. 查还原后的词 (family)
        if lemma in vocab_dict:
            rank = vocab_dict[lemma]
            if original != lemma:
                note = f"<{original}>" # 备注：原词是 families
        
        # 2. 如果还原后的词没查到，或者是生词(rank>20000)，再试试原词(families)
        # (防止 spaCy 还原错误，或者词库里只收录了变形体)
        elif original in vocab_dict:
            r_orig = vocab_dict[original]
            if r_orig < rank:
                rank = r_orig
                match = original
                note = ""

        processed_list.append({'单词': match, '排名': int(rank), '备注': note})

    # 转 DataFrame
    df_all = pd.DataFrame(processed_list)
    if df_all.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 分类
    known = df_all[df_all['排名'] <= r_start]
    target = df_all[(df_all['排名'] > r_start) & (df_all['排名'] <= r_end)]
    beyond = df_all[df_all['排名'] > r_end]
    
    return known, target, beyond

# ==========================================
# 4. 主界面
# ==========================================
text_input = st.text_area("在此粘贴文本:", height=150)

if st.button("🚀 开始分析 (spaCy Powered)", type="primary"):
    if not text_input.strip():
        st.warning("请输入内容")
    else:
        with st.spinner("spaCy 正在加载模型并分析..."):
            df_k, df_t, df_b = process_text(text_input)
        
        st.success("分析完成")
        t1, t2, t3 = st.tabs([
            f"🟡 重点 ({len(df_t)})", 
            f"🔴 超纲 ({len(df_b)})", 
            f"🟢 熟词 ({len(df_k)})"
        ])
        
        with t1: st.dataframe(df_t, use_container_width=True)
        with t2: st.dataframe(df_b, use_container_width=True)
        with t3: st.dataframe(df_k, use_container_width=True)
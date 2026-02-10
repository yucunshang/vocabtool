import streamlit as st
import pandas as pd
import re
import os
import spacy

# ==========================================
# 0. 核心引擎：加载 spaCy (工业级 NLP)
# ==========================================
st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🧠", layout="wide")

@st.cache_resource
def load_nlp():
    try:
        # 加载英语模型
        return spacy.load("en_core_web_sm")
    except OSError:
        # 如果通过链接安装失败，尝试直接下载（通常 requirements 写了链接不需要这步）
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

try:
    nlp = load_nlp()
    NLP_STATUS = "✅ spaCy 引擎就绪"
except Exception as e:
    st.error(f"spaCy 模型加载失败: {e}")
    st.stop()

def get_lemma_spacy(word):
    """
    使用 spaCy 进行精准还原
    families -> family
    are -> be
    went -> go
    """
    doc = nlp(word)
    # 取第一个词的 lemma_ (原形)
    return doc[0].lemma_.lower()

# ==========================================
# 1. 词库加载 (保持之前的稳健逻辑)
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
            df = pd.read_csv(file_path)
            if 'word' in df.columns and 'rank' in df.columns:
                vocab = pd.Series(df['rank'].values, index=df['word'].astype(str)).to_dict()
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
# 2. 界面显示与状态自检
# ==========================================
st.title("🧠 Vibe Vocab v10.0 (spaCy 版)")
st.caption("工业级 NLP 引擎 · 彻底解决变体识别问题")

if not vocab_dict:
    st.error(msg)
    st.stop()

# 侧边栏：状态面板
st.sidebar.success(NLP_STATUS)
st.sidebar.info(f"📚 词库: {msg}")

# === 关键自检区 ===
st.sidebar.markdown("---")
st.sidebar.markdown("**🔍 还原测试:**")
test_words = ["are", "went", "families", "better", "running"]
for t in test_words:
    res = get_lemma_spacy(t)
    st.sidebar.text(f"{t} -> {res}")
st.sidebar.markdown("*(如果 'are' 变成了 'be'，说明成功！)*")
# ===================

vocab_range = st.sidebar.slider("学习区间", 1, 20000, (6000, 8000), 500)
r_start, r_end = vocab_range

# ==========================================
# 3. 核心处理逻辑 (spaCy)
# ==========================================
def process_text(text):
    # 使用 spaCy 处理整个文本，它能根据上下文更精准地还原
    doc = nlp(text.lower())
    
    # 提取单词并去重
    # token.is_alpha 过滤掉标点和数字
    # token.lemma_ 直接拿到还原后的词
    
    seen_lemmas = set()
    unique_items = []
    
    for token in doc:
        if token.is_alpha and len(token.text) > 1:
            lemma = token.lemma_.lower()
            original = token.text.lower()
            
            # 排除停用词(如 the, is, a)的干扰，这里我们依靠词库排名来过滤
            # 但 spaCy 的 is_stop 也可以用，不过我们暂不开启，完全信任词库排名
            
            if lemma not in seen_lemmas:
                unique_items.append((original, lemma))
                seen_lemmas.add(lemma)
    
    # 排序以便查看
    unique_items.sort(key=lambda x: x[1])

    known, target, beyond = [], [], []
    
    for original, lemma in unique_items:
        rank = 99999
        match = lemma # 默认用还原后的词去查
        note = ""

        # 1. 查还原后的词 (be, family, go)
        if lemma in vocab_dict:
            rank = vocab_dict[lemma]
            if original != lemma:
                note = f"<{original}>"
        # 2. 兜底查原词 (有时候词库里收录的是 families 而不是 family)
        elif original in vocab_dict:
            r_orig = vocab_dict[original]
            if r_orig < rank:
                rank = r_orig
                match = original
                note = ""

        item = {'单词': match, '排名': int(rank), '备注': note}
        
        if rank <= r_start: known.append(item)
        elif r_start < rank <= r_end: target.append(item)
        else: beyond.append(item)

    return pd.DataFrame(known), pd.DataFrame(target), pd.DataFrame(beyond)

# ==========================================
# 4. 主界面
# ==========================================
text_input = st.text_area("在此粘贴文本:", height=150)

if st.button("🚀 开始分析 (spaCy Powered)", type="primary"):
    if not text_input.strip():
        st.warning("请输入内容")
    else:
        with st.spinner("spaCy 正在深度分析..."):
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
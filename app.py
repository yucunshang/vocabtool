import streamlit as st
import pandas as pd
import os
import sys
import subprocess

# ==========================================
# 1. Google Translate 风格配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Analyzer", page_icon="🅰️")

# 自定义 CSS：让界面更像 Google Translate (大文本框、清爽字体)
st.markdown("""
<style>
    /* 输入框样式 */
    .stTextArea textarea {
        font-size: 16px !important;
        font-family: 'Roboto', sans-serif;
        border-radius: 8px;
    }
    /* 数字输入框样式 */
    .stNumberInput input {
        font-weight: bold;
        color: #1a73e8; /* Google Blue */
    }
    /* 隐藏顶部多余的彩条 */
    header {visibility: hidden;}
    /* 调整顶部间距 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* 结果列表样式 */
    .vocab-list {
        font-family: monospace;
        font-size: 15px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心引擎 (spaCy + 自动修复)
# ==========================================
@st.cache_resource
def load_nlp():
    """加载或自动下载 spaCy 模型"""
    try:
        import spacy
    except ImportError:
        return None
    
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except:
        # 自动下载模型（防止报错）
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            return spacy.load(model_name)
        except:
            return None

# ==========================================
# 3. 词库加载 (静默加载)
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv"]

@st.cache_data
def load_vocab():
    file_path = next((f for f in POSSIBLE_FILES if os.path.exists(f)), None)
    if not file_path: return None
    
    try:
        df = pd.read_csv(file_path)
        # 极简清洗
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = cols
        
        # 智能找列
        w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
        r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1])
        
        df[w_col] = df[w_col].astype(str).str.lower().str.strip()
        df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
        
        return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
    except:
        return None

# 初始化资源
nlp = load_nlp()
vocab = load_vocab()

# ==========================================
# 4. 界面布局 (Top Bar + Split View)
# ==========================================

# --- 顶部：设置栏 ---
c1, c2, c3 = st.columns([1, 1, 3])
with c1:
    # 步长 500，默认 6000
    current_level = st.number_input("当前词汇量 (Current)", min_value=0, max_value=20000, value=6000, step=500)
with c2:
    # 步长 500，默认 8000
    target_level = st.number_input("目标词汇量 (Target)", min_value=0, max_value=20000, value=8000, step=500)
with c3:
    st.write("") # 占位

st.divider()

# --- 主体：左右分栏 ---
left_col, right_col = st.columns([1, 1])

# === 左侧：输入区 ===
with left_col:
    text_input = st.text_area(
        label="输入文本",
        placeholder="在此粘贴英语文章...",
        height=600,
        label_visibility="collapsed"
    )
    
    # 放在左侧底部的按钮
    analyze_btn = st.button("⚡ 开始分析 / Analyze", type="primary", use_container_width=True)

# === 右侧：结果区 ===
with right_col:
    if not nlp:
        st.error("正在初始化 NLP 引擎，请稍等或刷新...")
    elif not vocab:
        st.error("未找到词库文件 (coca_cleaned.csv)，请先上传。")
    elif analyze_btn and text_input:
        
        with st.spinner("Analyzing..."):
            # 1. spaCy 处理 (增加 max_length 防止大文本报错)
            nlp.max_length = 2000000 
            doc = nlp(text_input.lower())
            
            # 2. 提取与还原
            seen = set()
            data = []
            
            for token in doc:
                # 过滤非字母 (处理大小写、符号、非英文)
                if token.is_alpha and len(token.text) > 1:
                    lemma = token.lemma_ # 还原: families -> family
                    
                    if lemma not in seen:
                        # 查排名
                        rank = vocab.get(lemma, 99999)
                        
                        # 二次查找逻辑 (防止 spaCy 还原过度，或者词库里只有原词)
                        if rank == 99999 and token.text in vocab:
                            rank = vocab[token.text]
                            lemma = token.text
                            
                        data.append({'word': lemma, 'rank': int(rank)})
                        seen.add(lemma)
            
            # 3. 分组
            df = pd.DataFrame(data)
            
            if not df.empty:
                df = df.sort_values('rank')
                
                # 三个桶
                known = df[df['rank'] <= current_level]
                target = df[(df['rank'] > current_level) & (df['rank'] <= target_level)]
                beyond = df[df['rank'] > target_level]
                
                # 4. 显示结果 (Tabs)
                t1, t2, t3 = st.tabs([
                    f"🟡 重点 ({len(target)})", 
                    f"🔴 超纲 ({len(beyond)})", 
                    f"🟢 已掌握 ({len(known)})"
                ])
                
                # 定义纯文本渲染函数
                def render_text_list(dataframe):
                    if dataframe.empty:
                        st.caption("列表为空")
                        return
                    
                    # 生成文本列表: 1. word (1234)
                    lines = []
                    for i, row in dataframe.iterrows():
                        # 格式：单词 (排名)
                        lines.append(f"{row['word']} ({row['rank']})")
                    
                    # 使用滚动容器显示，防止页面过长
                    with st.container(height=500):
                        # join 换行符，直接显示纯文本
                        st.text("\n".join(lines))

                with t1:
                    render_text_list(target)
                with t2:
                    render_text_list(beyond)
                with t3:
                    render_text_list(known)
            else:
                st.warning("未检测到有效英文单词。")
                
    elif analyze_btn and not text_input:
        st.warning("请先在左侧粘贴文本。")
    else:
        # 空闲状态显示占位符
        st.info("👈 请在左侧输入文本，然后点击分析。")
        st.caption("支持大文本粘贴，系统会自动过滤符号和非英文内容。")
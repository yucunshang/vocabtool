import streamlit as st
import pandas as pd
import os
import sys
import subprocess

# ==========================================
# 1. Google Translate 风格配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Analyzer", page_icon="🅰️")

# 自定义 CSS 让界面更像 Google Translate (大文本框、清爽字体)
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 18px !important;
        line-height: 1.5 !important;
        font-family: 'Roboto', sans-serif;
    }
    .stNumberInput input {
        font-weight: bold;
        color: #1a73e8;
    }
    /* 隐藏部分多余的元素 */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心引擎 (spaCy + 自动修复)
# ==========================================
@st.cache_resource
def load_nlp():
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
# 3. 词库加载 (静默加载，不报错)
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
    current_level = st.number_input("当前词汇量", min_value=0, max_value=20000, value=6000, step=500)
with c2:
    # 步长 500，默认 8000
    target_level = st.number_input("目标词汇量", min_value=0, max_value=20000, value=8000, step=500)
with c3:
    st.write("") # 占位

# --- 主体：左右分栏 ---
st.divider()
left_col, right_col = st.columns([1, 1])

# === 左侧：输入区 ===
with left_col:
    st.markdown("### 📝 输入文本")
    text_input = st.text_area(
        label="hidden_label",
        placeholder="在此粘贴英语文章...",
        height=500,
        label_visibility="collapsed"
    )
    
    # 放在左侧底部的按钮
    analyze_btn = st.button("开始分析 / Analyze", type="primary", use_container_width=True)

# === 右侧：结果区 ===
with right_col:
    st.markdown("### 📊 分析结果")
    
    if not nlp:
        st.error("正在初始化 NLP 引擎，请稍等或刷新...")
    elif not vocab:
        st.error("未找到词库文件 (coca_cleaned.csv)，请先上传。")
    elif analyze_btn and text_input:
        
        with st.spinner("正在拆解文本..."):
            # 1. spaCy 处理 (增加 max_length 防止大文本报错)
            nlp.max_length = 2000000 
            doc = nlp(text_input.lower())
            
            # 2. 提取与还原
            seen = set()
            data = []
            
            for token in doc:
                if token.is_alpha and len(token.text) > 1:
                    lemma = token.lemma_ # 还原: families -> family
                    if lemma not in seen:
                        rank = vocab.get(lemma, 99999)
                        
                        # 二次查找逻辑 (防止 spaCy 还原过度)
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
                    f"🟡 重点词 ({len(target)})", 
                    f"🔴 超纲词 ({len(beyond)})", 
                    f"🟢 已掌握 ({len(known)})"
                ])
                
                # 定义一个简单的文本渲染函数
                def render_list(dataframe, color_code):
                    if dataframe.empty:
                        st.info("列表为空")
                        return
                    
                    # 生成简单的文本列表格式
                    # 格式: 1. word (Rank: 123)
                    lines = []
                    for _, row in dataframe.iterrows():
                        lines.append(f"• **{row['word']}** _(Rank: {row['rank']})_")
                    
                    # 使用 markdown 显示，带滚动条容器
                    with st.container(height=400):
                        st.markdown("\n".join(lines))

                with t1:
                    render_list(target, "orange")
                with t2:
                    render_list(beyond, "red")
                with t3:
                    render_list(known, "green")
            else:
                st.warning("未检测到有效英文单词。")
                
    elif analyze_btn and not text_input:
        st.warning("请先在左侧粘贴文本。")
    else:
        st.info("等待输入...")
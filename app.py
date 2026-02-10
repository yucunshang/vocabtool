import streamlit as st
import pandas as pd
import re
import os

# 设置页面
st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🧠", layout="wide")

# ==========================================
# 0. 核心引擎：延迟加载 spaCy 防止崩溃
# ==========================================
@st.cache_resource
def load_nlp():
    try:
        import spacy
        # 尝试加载模型
        try:
            return spacy.load("en_core_web_sm")
        except:
            # 如果模型没下成功，尝试这种方式加载
            import en_core_web_sm
            return en_core_web_sm.load()
    except ImportError:
        return None

nlp = load_nlp()

# ==========================================
# 1. 词库加载逻辑
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv"]

@st.cache_data
def load_vocab():
    file_path = next((f for f in POSSIBLE_FILES if os.path.exists(f)), None)
    if not file_path:
        return None, "❌ 未找到词库文件（coca_cleaned.csv）"
    try:
        df = pd.read_csv(file_path)
        # 清洗列名，防止 BOM 或空格干扰
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 确保有 word 和 rank 列
        w_col = 'word' if 'word' in df.columns else df.columns[0]
        r_col = 'rank' if 'rank' in df.columns else df.columns[1]
        
        # 统一格式：转小写，去空格
        df[w_col] = df[w_col].astype(str).str.lower().str.strip()
        
        # 构建字典
        vocab = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        return vocab, f"✅ 词库加载成功: {file_path}"
    except Exception as e:
        return None, f"❌ 读取失败: {str(e)}"

vocab_dict, status_msg = load_vocab()

# ==========================================
# 2. 侧边栏与 UI
# ==========================================
st.title("🧠 Vibe Vocab v12.0 (终极稳定版)")

if nlp is None:
    st.error("🚨 基础组件 (spaCy) 尚未安装成功。请确保 requirements.txt 已更新并点击 Manage app -> Reboot。")
    st.stop()

if not vocab_dict:
    st.error(status_msg)
    st.stop()

with st.sidebar:
    st.success("核心引擎已就绪")
    st.info(status_msg)
    st.divider()
    v_range = st.slider("设定学习区间", 1, 20000, (6000, 8000), 500)
    r_start, r_end = v_range
    st.write(f"🟢 熟词: 1-{r_start}")
    st.write(f"🟡 重点: {r_start}-{r_end}")
    st.write(f"🔴 超纲: {r_end}+")

# ==========================================
# 3. 核心处理逻辑
# ==========================================
def process_text_pro(text):
    # 使用 spaCy 进行全文本深度解析
    doc = nlp(text.lower())
    
    # 提取所有不重复的还原词 (Lemmas)
    results = []
    seen_lemmas = set()
    
    for token in doc:
        # 只处理长度 > 1 的纯字母单词
        if token.is_alpha and len(token.text) > 1:
            # 关键：使用 lemma_ 获取还原词（如 went -> go）
            lemma = token.lemma_.lower()
            original = token.text.lower()
            
            if lemma not in seen_lemmas:
                # 查词库
                rank = vocab_dict.get(lemma, 99999)
                
                # 特殊逻辑：如果还原词查不到，再试试原词（防漏）
                if rank == 99999 and original in vocab_dict:
                    rank = vocab_dict[original]
                    display_word = original
                else:
                    display_word = lemma
                
                results.append({
                    '单词': display_word,
                    '原文': original if original != display_word else "-",
                    '排名': int(rank)
                })
                seen_lemmas.add(lemma)

    # 排序并分类
    if not results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(results).sort_values('排名')
    known = df[df['排名'] <= r_start]
    target = df[(df['排名'] > r_start) & (df['排名'] <= r_end)]
    beyond = df[df['排名'] > r_end]
    return known, target, beyond

# ==========================================
# 4. 主界面交互
# ==========================================
text_input = st.text_area("在此粘贴你的英文文章/小说内容:", height=200)

if st.button("🚀 开始精准分析", type="primary"):
    if not text_input.strip():
        st.warning("请输入文本内容")
    else:
        with st.spinner("正在进行工业级词形还原分析..."):
            df_k, df_t, df_b = process_text_pro(text_input)
        
        st.success(f"分析完成！找到重点词: {len(df_t)} 个")
        
        tab1, tab2, tab3 = st.tabs([
            f"🟡 重点突破 ({len(df_t)})", 
            f"🔴 生词/超纲 ({len(df_b)})", 
            f"🟢 熟词表 ({len(df_k)})"
        ])
        
        with tab1:
            st.dataframe(df_t, use_container_width=True)
            if not df_t.empty:
                csv_t = df_t.to_csv(index=False).encode('utf-8')
                st.download_button("📥 下载重点词 CSV", csv_t, "target.csv", "text/csv")
            
        with tab2:
            st.dataframe(df_b, use_container_width=True)
            
        with tab3:
            st.dataframe(df_k, use_container_width=True)
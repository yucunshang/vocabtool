import streamlit as st
import pandas as pd
import re
import os
from io import BytesIO

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="⚡", layout="wide")

# --- 1. 配置与加载 ---
DEFAULT_VOCAB_FILE = "coca_cleaned.csv" 

@st.cache_data
def load_vocab():
    """读取内置词库"""
    if not os.path.exists(DEFAULT_VOCAB_FILE):
        return None
    try:
        df = pd.read_csv(DEFAULT_VOCAB_FILE)
        # 标准化列名
        df.columns = [c.strip().lower() for c in df.columns]
        # 建立字典加速查找: word -> rank
        if 'word' in df.columns and 'rank' in df.columns:
            return pd.Series(df['rank'].values, index=df['word'].astype(str)).to_dict()
        else:
            return None
    except:
        return None

# --- 2. 核心逻辑 (纯净版) ---
def process_text_pure(text, vocab_dict, user_limit):
    # 转小写
    text_lower = text.lower()
    # 正则提取单词 (至少2个字母)
    words = re.findall(r'\b[a-z]{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    unknown_list = []
    known_list = []
    
    for w in unique_words:
        rank = 999999
        match_word = w
        
        # 查词逻辑
        if w in vocab_dict:
            rank = vocab_dict[w]
        elif w.endswith('s') and w[:-1] in vocab_dict:
            match_word = w[:-1]
            rank = vocab_dict[match_word]
        elif w.endswith('ed') and w[:-2] in vocab_dict:
            match_word = w[:-2]
            rank = vocab_dict[match_word]
        elif w.endswith('ing') and w[:-3] in vocab_dict:
            match_word = w[:-3]
            rank = vocab_dict[match_word]
            
        # 分组
        item = {'单词 (Word)': match_word, '排名 (Rank)': rank}
        
        if rank <= user_limit:
            known_list.append(item)
        else:
            unknown_list.append(item)

    # 转为 DataFrame 并按排名排序
    df_unknown = pd.DataFrame(unknown_list)
    if not df_unknown.empty:
        df_unknown = df_unknown.sort_values('排名 (Rank)')
        
    df_known = pd.DataFrame(known_list)
    if not df_known.empty:
        df_known = df_known.sort_values('排名 (Rank)')
        
    return df_unknown, df_known

# --- 3. 界面 UI ---
st.title("⚡ Vibe Vocab Studio")
st.caption("纯净版：无上下文 · 极速分析 · 双格式下载")

# 加载数据
vocab_dict = load_vocab()

if vocab_dict is None:
    st.error(f"❌ 错误：找不到 {DEFAULT_VOCAB_FILE}，请确保已上传该文件到 GitHub！")
    st.stop()

# 侧边栏
st.sidebar.header("⚙️ 设置")
st.sidebar.success("✅ 词库已就绪")
user_vocab = st.sidebar.slider("你的词汇量阈值", 1000, 20000, 6000, 500)

# 输入区
with st.expander("📝 文本输入", expanded=True):
    tab_paste, tab_upload = st.tabs(["粘贴文本", "上传 TXT"])
    with tab_paste:
        text_input = st.text_area("在此粘贴内容:", height=150)
    with tab_upload:
        uploaded = st.file_uploader("上传文件", type="txt")
        if uploaded:
            text_input = uploaded.read().decode("utf-8")

final_text = text_input if text_input else ""

# 分析按钮
if st.button("🚀 开始分析", type="primary"):
    if not final_text.strip():
        st.warning("请先输入文本！")
    else:
        # 执行分析
        unk_df, kn_df = process_text_pure(final_text, vocab_dict, user_vocab)
        
        st.success(f"分析完成！生词: {len(unk_df)} | 熟词: {len(kn_df)}")
        
        # 结果展示区
        tab1, tab2 = st.tabs([f"🔴 生词表 ({len(unk_df)})", f"🟢 熟词表 ({len(kn_df)})"])
        
        # --- 生词 Tab ---
        with tab1:
            if not unk_df.empty:
                st.dataframe(unk_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                # 下载 CSV
                csv_unk = unk_df.to_csv(index=False).encode('utf-8')
                col1.download_button("📥 下载 CSV (Excel)", csv_unk, "unknown_words.csv", "text/csv")
                
                # 下载 TXT (只包含单词，一行一个，方便导入背单词软件)
                txt_unk = "\n".join(unk_df['单词 (Word)'].tolist())
                col2.download_button("📄 下载 TXT (纯单词)", txt_unk, "unknown_words.txt", "text/plain")
            else:
                st.info("太棒了！没有发现生词。")

        # --- 熟词 Tab ---
        with tab2:
            if not kn_df.empty:
                st.dataframe(kn_df, use_container_width=True)
                
                col3, col4 = st.columns(2)
                # 下载 CSV
                csv_kn = kn_df.to_csv(index=False).encode('utf-8')
                col3.download_button("📥 下载 CSV (Excel)", csv_kn, "known_words.csv", "text/csv")
                
                # 下载 TXT
                txt_kn = "\n".join(kn_df['单词 (Word)'].tolist())
                col4.download_button("📄 下载 TXT (纯单词)", txt_kn, "known_words.txt", "text/plain")
            else:
                st.info("没有发现熟词（可能是阈值设置太低？）")
import streamlit as st
import pandas as pd
import re
import os
from io import BytesIO

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🎯", layout="wide")

# --- 1. 智能加载配置 ---
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]

@st.cache_data
def load_vocab():
    file_path = None
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            file_path = f
            break
            
    if not file_path:
        return None

    try:
        try:
            df = pd.read_csv(file_path)
        except:
            df = pd.read_csv(file_path, encoding='gbk')

        df.columns = [str(c).strip().lower().replace('\n', '') for c in df.columns]
        
        rank_col = None
        for c in df.columns:
            if any(k in c for k in ['rank', '排名', '序号', '词频']):
                rank_col = c
                break
        
        word_col = None
        for c in df.columns:
            if any(k in c for k in ['word', '单词', '词汇']):
                word_col = c
                break
        
        if not word_col: word_col = df.columns[0]
        if not rank_col: rank_col = df.columns[3] if len(df.columns) > 3 else df.columns[0]

        vocab_dict = pd.Series(
            pd.to_numeric(df[rank_col], errors='coerce').fillna(99999).values, 
            index=df[word_col].astype(str).str.lower().str.strip()
        ).to_dict()
        
        return vocab_dict
    except Exception as e:
        st.error(f"词库加载出错: {e}")
        return None

# --- 2. 三段式核心逻辑 ---
def process_text_three_tiers(text, vocab_dict, range_start, range_end):
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    tier_known = []   
    tier_target = []  
    tier_beyond = []  
    
    for w in unique_words:
        rank = 999999
        match_word = w
        
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
            
        item = {'单词 (Word)': match_word, '排名 (Rank)': int(rank)}
        
        if rank <= range_start:
            tier_known.append(item)
        elif range_start < rank <= range_end:
            tier_target.append(item)
        else:
            tier_beyond.append(item)

    df_known = pd.DataFrame(tier_known).sort_values('排名 (Rank)') if tier_known else pd.DataFrame()
    df_target = pd.DataFrame(tier_target).sort_values('排名 (Rank)') if tier_target else pd.DataFrame()
    df_beyond = pd.DataFrame(tier_beyond).sort_values('排名 (Rank)') if tier_beyond else pd.DataFrame()
        
    return df_known, df_target, df_beyond

# --- 3. 界面 UI ---
st.title("🎯 Vibe Vocab v3.0 (分级突击版)") # 注意这里，我看能不能更新成功
st.caption("自定义学习区间 · 精准锁定目标词汇")

vocab_dict = load_vocab()
if not vocab_dict:
    st.error("❌ 找不到词库文件！请确认 GitHub 仓库里有 csv 文件。")
    st.stop()

st.sidebar.header("⚙️ 学习规划")
st.sidebar.success(f"📚 词库已加载 ({len(vocab_dict)}词)")

st.sidebar.subheader("设定你的范围")
vocab_range = st.sidebar.slider(
    "拖动滑块选择区间：",
    min_value=1, 
    max_value=20000, 
    value=(6000, 8000), 
    step=500
)

range_start = vocab_range[0]
range_end = vocab_range[1]

st.sidebar.info(
    f"🟢 **熟词**: 1 - {range_start}\n\n"
    f"🟡 **重点**: {range_start} - {range_end}\n\n"
    f"🔴 **超纲**: {range_end}+"
)

with st.expander("📝 文本输入", expanded=True):
    tab_paste, tab_upload = st.tabs(["粘贴文本", "上传 TXT"])
    with tab_paste:
        text_input = st.text_area("在此粘贴:", height=150)
    with tab_upload:
        uploaded = st.file_uploader("上传文件", type="txt")
        if uploaded:
            text_input = uploaded.read().decode("utf-8")

final_text = text_input if text_input else ""

def show_download_buttons(df, prefix):
    if df.empty: return
    col1, col2 = st.columns(2)
    csv = df.to_csv(index=False).encode('utf-8')
    col1.download_button(f"📥 下载 {prefix} Excel", csv, f"{prefix}.csv", "text/csv")
    txt = "\n".join(df['单词 (Word)'].tolist())
    col2.download_button(f"📄 下载 {prefix} TXT", txt, f"{prefix}.txt", "text/plain")

if st.button("🚀 开始精准分析", type="primary"):
    if not final_text.strip():
        st.warning("请先输入文本！")
    else:
        df_known, df_target, df_beyond = process_text_three_tiers(final_text, vocab_dict, range_start, range_end)
        
        st.success(f"分析完成！ 重点目标词汇: {len(df_target)} 个")
        
        t1, t2, t3 = st.tabs([
            f"🟡 重点突破 ({len(df_target)})", 
            f"🔴 超纲/生词 ({len(df_beyond)})", 
            f"🟢 已掌握 ({len(df_known)})"
        ])
        
        with t1:
            st.markdown(f"### 🎯 你的核心学习区 ({range_start}-{range_end})")
            if not df_target.empty:
                st.dataframe(df_target, use_container_width=True)
                show_download_buttons(df_target, "target_words")
            else:
                st.info("太棒了，没有发现这个范围的词！")

        with t2:
            st.markdown(f"### 🚀 暂时跳过的难词 (>{range_end})")
            if not df_beyond.empty:
                st.dataframe(df_beyond, use_container_width=True)
                show_download_buttons(df_beyond, "beyond_words")
            else:
                st.info("没有超纲词汇！")

        with t3:
            st.markdown(f"### ✅ 无需复习的熟词 (<{range_start})")
            if not df_known.empty:
                st.dataframe(df_known, use_container_width=True)
                show_download_buttons(df_known, "known_words")
            else:
                st.info("没有发现熟词。")
import streamlit as st
import pandas as pd
import re
import os
from io import BytesIO

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🎯", layout="wide")

# --- 1. 智能加载配置 ---
# 优先找清洗过的文件，找不到就找原始名，方便用户偷懒
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]

@st.cache_data
def load_vocab():
    """读取并智能识别词库"""
    file_path = None
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            file_path = f
            break
            
    if not file_path:
        return None

    try:
        # 读取文件
        try:
            df = pd.read_csv(file_path)
        except:
            df = pd.read_csv(file_path, encoding='gbk') # 尝试gbk防乱码

        # === 智能列名映射 (关键更新) ===
        # 把所有列名转小写、去空格
        df.columns = [str(c).strip().lower().replace('\n', '') for c in df.columns]
        
        # 寻找 Rank 列 (支持中文)
        rank_col = None
        for c in df.columns:
            if any(k in c for k in ['rank', '排名', '序号', '词频']):
                rank_col = c
                break
        
        # 寻找 Word 列 (支持中文)
        word_col = None
        for c in df.columns:
            if any(k in c for k in ['word', '单词', '词汇']):
                word_col = c
                break
        
        # 兜底：如果找不到，盲猜第1列和第4列(根据你的文件结构)
        if not word_col: word_col = df.columns[0]
        if not rank_col: rank_col = df.columns[3] if len(df.columns) > 3 else df.columns[0]

        # 建立字典: word -> rank
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
    # 清洗文本
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    # 初始化三个桶
    tier_known = []   # < Start (已掌握)
    tier_target = []  # Start ~ End (重点突破)
    tier_beyond = []  # > End (超纲)
    
    for w in unique_words:
        rank = 999999
        match_word = w
        
        # 匹配变体 (s, ed, ing)
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
        
        # === 三级分拣 ===
        if rank <= range_start:
            tier_known.append(item)
        elif range_start < rank <= range_end:
            tier_target.append(item)
        else:
            tier_beyond.append(item)

    # 转 DataFrame
    df_known = pd.DataFrame(tier_known).sort_values('排名 (Rank)') if tier_known else pd.DataFrame()
    df_target = pd.DataFrame(tier_target).sort_values('排名 (Rank)') if tier_target else pd.DataFrame()
    df_beyond = pd.DataFrame(tier_beyond).sort_values('排名 (Rank)') if tier_beyond else pd.DataFrame()
        
    return df_known, df_target, df_beyond

# --- 3. 界面 UI ---
st.title("🎯 Vibe Vocab - 分级突击版")
st.caption("自定义学习区间 · 精准锁定目标词汇")

# 加载
vocab_dict = load_vocab()
if not vocab_dict:
    st.error("❌ 找不到词库文件！请确认 GitHub 仓库里有 csv 文件。")
    st.stop()

# 侧边栏
st.sidebar.header("⚙️ 学习规划")
st.sidebar.success(f"📚 词库已加载 ({len(vocab_dict)}词)")

# === 关键更新：双滑块 ===
st.sidebar.subheader("设定你的范围")
# 默认 6000-8000
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

# 输入区
with st.expander("📝 文本输入", expanded=True):
    tab_paste, tab_upload = st.tabs(["粘贴文本", "上传 TXT"])
    with tab_paste:
        text_input = st.text_area("在此粘贴:", height=150)
    with tab_upload:
        uploaded = st.file_uploader("上传文件", type="txt")
        if uploaded:
            text_input = uploaded.read().decode("utf-8")

final_text = text_input if text_input else ""

# 结果展示函数
def show_download_buttons(df, prefix):
    if df.empty: return
    col1, col2 = st.columns(2)
    # CSV
    csv = df.to_csv(index=False).encode('utf-8')
    col1.download_button(f"📥 下载 {prefix} Excel", csv, f"{prefix}.csv", "text/csv")
    # TXT
    txt = "\n".join(df['单词 (Word)'].tolist())
    col2.download_button(f"📄 下载 {prefix} TXT", txt, f"{prefix}.txt", "text/plain")

if st.button("🚀 开始精准分析", type="primary"):
    if not final_text.strip():
        st.warning("请先输入文本！")
    else:
        df_known, df_target, df_beyond = process_text_three_tiers(final_text, vocab_dict, range_start, range_end)
        
        st.success(f"分析完成！ 重点目标词汇: {len(df_target)} 个")
        
        # 三个标签页
        t1, t2, t3 = st.tabs([
            f"🟡 重点突破 ({len(df_target)})", 
            f"🔴 超纲/生词 ({len(df_beyond)})", 
            f"🟢 已掌握 ({len(df_known)})"
        ])
        
        # Tab 1: 重点 (最重要，放第一个)
        with t1:
            st.markdown(f"### 🎯 你的核心学习区 ({range_start}-{range_end})")
            if not df_target.empty:
                st.dataframe(df_target, use_container_width=True)
                show_download_buttons(df_target, "target_words")
            else:
                st.info("这篇文章里没有这个范围的词，太棒了（或者文章太简单/太难）！")

        # Tab 2: 超纲
        with t2:
            st.markdown(f"### 🚀 暂时跳过的难词 (>{range_end})")
            if not df_beyond.empty:
                st.dataframe(df_beyond, use_container_width=True)
                show_download_buttons(df_beyond, "beyond_words")
            else:
                st.info("没有超纲词汇！")

        # Tab 3: 熟词
        with t3:
            st.markdown(f"### ✅ 无需复习的熟词 (<{range_start})")
            if not df_known.empty:
                st.dataframe(df_known, use_container_width=True)
                show_download_buttons(df_known, "known_words")
            else:
                st.info("没有发现熟词。")
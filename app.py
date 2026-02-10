import streamlit as st
import pandas as pd
import re
import os
import simplemma

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🧠", layout="wide")

# --- 1. 自动适配 Simplemma (保持不变) ---
try:
    simplemma.lemmatize("t", lang="en")
    def get_lemma(word): return simplemma.lemmatize(word, lang="en")
except TypeError:
    if hasattr(simplemma, 'load_data'):
        lang_data = simplemma.load_data('en')
        def get_lemma(word): return simplemma.lemmatize(word, lang_data)
    else:
        def get_lemma(word): return word 

# --- 2. 智能分流加载 (核心修复) ---
# 优先读取 coca_cleaned.csv
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]

@st.cache_data
def load_vocab():
    file_path = None
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            file_path = f
            break
            
    if not file_path: return None, "未找到文件"

    try:
        # 尝试读取 (优先 utf-8-sig 去除 BOM 头)
        df = None
        for enc in ['utf-8-sig', 'utf-8', 'gbk']:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                if len(df) > 10: break
            except: continue
        
        if df is None: return None, "读取失败"

        # === 核心修复逻辑 ===
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = cols # 重命名列名以便查找

        word_col = None
        rank_col = None

        # 情况 A: 清洗过的文件 (通常只有 word, rank 两列)
        if 'rank' in cols and 'word' in cols:
            word_col = 'word'
            rank_col = 'rank'
        # 情况 B: 只有两列，且列名不对 (盲猜)
        elif len(cols) == 2:
            # 假设第1列是词，第2列是排名(数字)
            word_col = df.columns[0]
            rank_col = df.columns[1]
        # 情况 C: 原始乱文件 (多列)
        elif len(cols) >= 4:
            # 原始文件第1列是单词，第4列(索引3)是排名
            word_col = df.columns[0]
            rank_col = df.columns[3]
        
        # 如果还是没找到，尝试关键词搜索
        if not rank_col:
            rank_col = next((c for c in df.columns if any(k in c for k in ['rank', '排名', '序号', '词频'])), None)
        if not word_col:
            word_col = next((c for c in df.columns if any(k in c for k in ['word', '单词', '词汇'])), None)

        if not word_col or not rank_col:
            return None, f"无法识别列名。检测到的列: {cols}"

        # === 数据清洗 ===
        # 强制转小写，去空格
        df['word_clean'] = df[word_col].astype(str).str.lower().str.strip()
        # 强制转数字
        df['rank_clean'] = pd.to_numeric(df[rank_col], errors='coerce').fillna(99999)
        
        # 再次过滤：确保 rank 是有效数字且 > 0
        df = df[df['rank_clean'] > 0]
        df = df[df['rank_clean'] < 99999]

        vocab_dict = pd.Series(
            df['rank_clean'].values, 
            index=df['word_clean']
        ).to_dict()
        
        return vocab_dict, f"已加载: {file_path} (单词列:{word_col}, 排名列:{rank_col})"

    except Exception as e:
        return None, str(e)

# --- 3. 核心处理逻辑 ---
def process_text_smart(text, vocab_dict, range_start, range_end):
    text_lower = text.lower()
    words = re.findall(r'\b[a-z\']{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    tier_known = []   
    tier_target = []  
    tier_beyond = []  
    
    for w in unique_words:
        rank = 999999
        match_word = w
        
        # 1. 查原形
        if w in vocab_dict:
            rank = vocab_dict[w]
            match_word = w
        else:
            # 2. 查还原
            lemma = get_lemma(w)
            if lemma in vocab_dict:
                rank = vocab_dict[lemma]
                match_word = lemma
            else:
                # 3. 查变体
                if w.endswith("'s") and w[:-2] in vocab_dict:
                    rank = vocab_dict[w[:-2]]
                    match_word = w[:-2]

        item = {'单词': match_word, '原文': w, '排名': int(rank)}
        
        if rank <= range_start:
            tier_known.append(item)
        elif range_start < rank <= range_end:
            tier_target.append(item)
        else:
            if item['单词'] == item['原文']:
                item['原文'] = '-'
            tier_beyond.append(item)

    def to_df(data):
        if not data: return pd.DataFrame()
        return pd.DataFrame(data).sort_values('排名').drop_duplicates(subset=['单词'])

    return to_df(tier_known), to_df(tier_target), to_df(tier_beyond)

# --- 4. 界面 UI ---
st.title("🧠 Vibe Vocab v5.2 (智能双核版)")
st.caption("完美适配 coca_cleaned.csv")

vocab_dict, status_msg = load_vocab()

# 侧边栏显示加载状态，方便调试
st.sidebar.header("⚙️ 系统状态")
if vocab_dict:
    st.sidebar.success(f"✅ 成功! {len(vocab_dict)}词")
    st.sidebar.caption(f"详情: {status_msg}")
else:
    st.sidebar.error("❌ 词库加载失败")
    st.sidebar.code(status_msg)
    st.stop()

st.sidebar.subheader("设定学习区间")
vocab_range = st.sidebar.slider(
    "拖动滑块：", 1, 20000, (6000, 8000), 500
)
range_start, range_end = vocab_range

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
    col1.download_button(f"📥 下载 Excel", csv, f"{prefix}.csv", "text/csv")
    txt = "\n".join(df['单词'].tolist())
    col2.download_button(f"📄 下载 TXT", txt, f"{prefix}.txt", "text/plain")

if st.button("🚀 开始分析", type="primary"):
    if not final_text.strip():
        st.warning("请先输入文本！")
    else:
        df_known, df_target, df_beyond = process_text_smart(final_text, vocab_dict, range_start, range_end)
        
        st.success(f"分析完成！")
        
        t1, t2, t3 = st.tabs([
            f"🟡 重点突破 ({len(df_target)})", 
            f"🔴 超纲/生词 ({len(df_beyond)})", 
            f"🟢 已掌握 ({len(df_known)})"
        ])
        
        with t1:
            st.markdown(f"### 🎯 重点学习 ({range_start}-{range_end})")
            if not df_target.empty:
                st.dataframe(df_target, use_container_width=True)
                show_download_buttons(df_target, "target_words")
            else:
                st.info("此区间无生词。")

        with t2:
            st.markdown(f"### 🚀 超纲词 (>{range_end})")
            if not df_beyond.empty:
                st.dataframe(df_beyond, use_container_width=True)
                show_download_buttons(df_beyond, "beyond_words")
            else:
                st.info("没有超纲词。")

        with t3:
            st.markdown(f"### ✅ 已掌握 (<{range_start})")
            if not df_known.empty:
                st.dataframe(df_known, use_container_width=True)
                show_download_buttons(df_known, "known_words")
            else:
                st.info("无熟词。")
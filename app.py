import streamlit as st
import pandas as pd
import re
import os
import simplemma

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🧠", layout="wide")

# --- 1. 自动适配 Simplemma 版本 ---
try:
    test_res = simplemma.lemmatize("testing", lang="en")
    def get_lemma(word):
        return simplemma.lemmatize(word, lang="en")
except TypeError:
    if hasattr(simplemma, 'load_data'):
        lang_data = simplemma.load_data('en')
        def get_lemma(word):
            return simplemma.lemmatize(word, lang_data)
    else:
        def get_lemma(word):
            return word 

# --- 2. 强力加载词库 (关键修复) ---
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
        # 1. 尝试读取 (加强编码兼容性)
        df = None
        for enc in ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                # 如果成功读出多列，说明编码对了
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        if df is None: return None

        # 2. 暴力锁定列 (不再依赖列名)
        # 你的文件结构：Column 0 是单词，Column 3 是排名
        if len(df.columns) >= 4:
            word_col = df.columns[0] # 第1列
            rank_col = df.columns[3] # 第4列
        else:
            # 兜底：如果用户换了文件，尝试智能查找
            df.columns = [str(c).strip().lower().replace('\n', '') for c in df.columns]
            rank_col = next((c for c in df.columns if any(k in c for k in ['rank', '排名', '序号', '词频'])), df.columns[0])
            word_col = next((c for c in df.columns if any(k in c for k in ['word', '单词', '词汇'])), df.columns[1])

        # 3. 建立字典 (清洗数据)
        # 强制把排名转为数字，无法转换的(比如表头)变NaN然后填充99999
        df['rank_clean'] = pd.to_numeric(df[rank_col], errors='coerce').fillna(99999)
        df['word_clean'] = df[word_col].astype(str).str.lower().str.strip()
        
        # 过滤掉无效行
        df = df[df['rank_clean'] < 99990] 
        
        vocab_dict = pd.Series(
            df['rank_clean'].values, 
            index=df['word_clean']
        ).to_dict()
        
        return vocab_dict
    except Exception as e:
        st.error(f"词库加载出错: {e}")
        return None

# --- 3. 核心逻辑 ---
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
st.title("🧠 Vibe Vocab v5.1 (强力修复版)")
st.caption("强制列对齐 · 解决简单词排名错误")

vocab_dict = load_vocab()
if not vocab_dict:
    st.error("❌ 找不到词库或读取失败！")
    st.stop()

st.sidebar.header("⚙️ 学习规划")
st.sidebar.success(f"📚 词库加载成功: {len(vocab_dict)} 词")

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

if st.button("🚀 开始智能分析", type="primary"):
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
            # 调试信息：如果这里出现了简单的词，说明词库没读对
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
import streamlit as st
import pandas as pd
import re
import os
import simplemma

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🧠", layout="wide")

# --- 1. 智能加载配置 ---
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]
LANG_DATA = simplemma.load_data('en') # 加载英语还原库

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

        # 清洗列名
        df.columns = [str(c).strip().lower().replace('\n', '') for c in df.columns]
        
        # 智能找列
        rank_col = next((c for c in df.columns if any(k in c for k in ['rank', '排名', '序号', '词频'])), None)
        word_col = next((c for c in df.columns if any(k in c for k in ['word', '单词', '词汇'])), None)
        
        if not word_col: word_col = df.columns[0]
        if not rank_col: rank_col = df.columns[3] if len(df.columns) > 3 else df.columns[0]

        # 建立字典: word -> rank
        # 关键优化：把词库里的词也都做一次还原，确保命中率
        vocab_dict = pd.Series(
            pd.to_numeric(df[rank_col], errors='coerce').fillna(99999).values, 
            index=df[word_col].astype(str).str.lower().str.strip()
        ).to_dict()
        
        return vocab_dict
    except Exception as e:
        st.error(f"词库加载出错: {e}")
        return None

# --- 2. 核心逻辑 (v4.0 强力还原版) ---
def process_text_smart(text, vocab_dict, range_start, range_end):
    text_lower = text.lower()
    # 提取单词
    words = re.findall(r'\b[a-z\']{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    tier_known = []   
    tier_target = []  
    tier_beyond = []  
    
    for w in unique_words:
        # === 核心修改：三级跳查词法 ===
        rank = 999999
        match_word = w
        
        # 1. 查原词 (比如 "apple")
        if w in vocab_dict:
            rank = vocab_dict[w]
            match_word = w
        else:
            # 2. 查还原后的词 (比如 "went" -> "go", "countries" -> "country")
            lemma = simplemma.lemmatize(w, LANG_DATA)
            if lemma in vocab_dict:
                rank = vocab_dict[lemma]
                match_word = lemma # 修正显示为原形，方便学习
            else:
                # 3. 最后的挣扎 (去掉 's 等)
                if w.endswith("'s") and w[:-2] in vocab_dict:
                    rank = vocab_dict[w[:-2]]
                    match_word = w[:-2]

        item = {'单词 (Word)': match_word, '原文 (Original)': w, '排名 (Rank)': int(rank)}
        
        # 分级
        if rank <= range_start:
            tier_known.append(item)
        elif range_start < rank <= range_end:
            tier_target.append(item)
        else:
            # 如果原文和还原后是一样的，只显示一个列
            if item['单词 (Word)'] == item['原文 (Original)']:
                item['原文 (Original)'] = '-'
            tier_beyond.append(item)

    # 结果生成
    def to_df(data):
        if not data: return pd.DataFrame()
        return pd.DataFrame(data).sort_values('排名 (Rank)').drop_duplicates(subset=['单词 (Word)'])

    return to_df(tier_known), to_df(tier_target), to_df(tier_beyond)

# --- 3. 界面 UI ---
st.title("🧠 Vibe Vocab v4.0 (智能还原版)")
st.caption("Simplemma 驱动 · 完美处理变体/不规则动词")

vocab_dict = load_vocab()
if not vocab_dict:
    st.error("❌ 找不到词库文件！")
    st.stop()

st.sidebar.header("⚙️ 学习规划")
st.sidebar.success(f"📚 词库已就绪")

st.sidebar.subheader("设定范围")
vocab_range = st.sidebar.slider(
    "选择区间：", 1, 20000, (6000, 8000), 500
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
    txt = "\n".join(df['单词 (Word)'].tolist())
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
            st.markdown(f"### 🎯 你的核心学习区 ({range_start}-{range_end})")
            if not df_target.empty:
                st.dataframe(df_target, use_container_width=True)
                show_download_buttons(df_target, "target_words")
            else:
                st.info("没有发现此区间的单词。")

        with t2:
            st.markdown(f"### 🚀 超纲词 (>{range_end})")
            # 超纲词往往是还原失败的，或者真的很偏
            if not df_beyond.empty:
                st.dataframe(df_beyond, use_container_width=True)
                show_download_buttons(df_beyond, "beyond_words")
            else:
                st.info("没有超纲词汇！")

        with t3:
            st.markdown(f"### ✅ 已掌握 (<{range_start})")
            if not df_known.empty:
                st.dataframe(df_known, use_container_width=True)
                show_download_buttons(df_known, "known_words")
            else:
                st.info("没有发现熟词。")
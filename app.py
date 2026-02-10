import streamlit as st
import pandas as pd
import re
import os

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="⚡", layout="wide")

# --- 核心配置 ---
# 这里写死文件名，因为你已经把它传到 GitHub 了
DEFAULT_VOCAB_FILE = "coca_cleaned.csv" 

# --- 核心逻辑 ---
def get_sentence_context(text, word):
    """提取原句"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if re.search(r'\b' + re.escape(word) + r'\b', sent, re.IGNORECASE):
            return sent.strip()[:300]
    return "未找到原句"

@st.cache_data
def load_vocab():
    """自动加载内置词库"""
    if not os.path.exists(DEFAULT_VOCAB_FILE):
        return None
    try:
        # 读取 CSV，标准化列名
        df = pd.read_csv(DEFAULT_VOCAB_FILE)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"内置词库读取失败: {e}")
        return None

def process_text_lite(text, vocab_df, user_limit):
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    # 建立字典加速
    if 'word' in vocab_df.columns and 'rank' in vocab_df.columns:
        vocab_dict = pd.Series(vocab_df['rank'].values, index=vocab_df['word'].astype(str)).to_dict()
    else:
        return pd.DataFrame(), pd.DataFrame()

    found_items = []
    for w in unique_words:
        rank = 999999
        match_word = w
        
        # 匹配逻辑
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
            
        is_unknown = rank > user_limit
        context = get_sentence_context(text, w) if is_unknown else ""
            
        found_items.append({
            'word': match_word,
            'rank': rank,
            'is_known': not is_unknown,
            'context': context
        })

    df = pd.DataFrame(found_items)
    if not df.empty:
        return df[~df['is_known']].sort_values('rank'), df[df['is_known']].sort_values('rank')
    return pd.DataFrame(), pd.DataFrame()

# --- 界面 UI ---
st.title("⚡ Vibe Vocab Studio")
st.caption("内置 COCA 20000 词表 · 自动分级 · Anki 制卡")

# 加载数据
vocab_df = load_vocab()

if vocab_df is None:
    st.error(f"❌ 错误：在仓库中找不到 {DEFAULT_VOCAB_FILE} 文件！请确认你已经上传了该文件。")
    st.stop()

# 侧边栏 (简化了，不需要上传文件)
st.sidebar.header("⚙️ 设置")
st.sidebar.success("✅ 内置词库已加载")
user_vocab = st.sidebar.slider("你的词汇量阈值", 1000, 20000, 6000, 500)

# 输入区
with st.expander("📝 文本输入 (支持长文本)", expanded=True):
    tab1, tab2 = st.tabs(["粘贴文本", "上传文件"])
    with tab1:
        text_input_raw = st.text_area("在此粘贴:", height=150)
    with tab2:
        uploaded_txt = st.file_uploader("上传 .txt 小说/文章", type="txt")
        if uploaded_txt:
            text_input_raw = uploaded_txt.read().decode("utf-8")

final_text = text_input_raw if text_input_raw else ""

if st.button("🚀 开始分析", type="primary"):
    if not final_text.strip():
        st.warning("请输入文本内容！")
    else:
        with st.spinner("正在分析..."):
            unknown_df, known_df = process_text_lite(final_text, vocab_df, user_vocab)
        
        st.success(f"分析完成！发现 {len(unknown_df)} 个生词。")
        
        tab_unk, tab_kn, tab_anki = st.tabs(["🔴 生词表", "🟢 熟词表", "🎴 Anki 制卡"])
        
        with tab_unk:
            st.dataframe(unknown_df[['word', 'rank', 'context']], use_container_width=True)
            csv = unknown_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 下载生词 CSV", csv, "unknown.csv", "text/csv")

        with tab_kn:
            st.dataframe(known_df[['word', 'rank']], use_container_width=True)

        with tab_anki:
            st.info("已自动生成 Anki 导入格式 (正面:单词 | 背面:原句+排名)")
            if not unknown_df.empty:
                anki_df = pd.DataFrame()
                anki_df['Front'] = unknown_df['word']
                anki_df['Back'] = unknown_df['context'] + "<br><br>Rank: #" + unknown_df['rank'].astype(str)
                
                anki_csv = anki_df.to_csv(index=False, header=False).encode('utf-8')
                st.download_button("⚡ 下载 Anki 牌组", anki_csv, "anki_deck.csv", "text/csv")
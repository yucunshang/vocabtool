import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="⚡", layout="wide")

# --- 核心逻辑：不依赖 AI，纯算法极速处理 ---

def get_sentence_context(text, word):
    """用正则查找包含单词的句子 (替代 Spacy)"""
    # 简单的分句逻辑：按 . ! ? 分割
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        # 如果单词(作为独立词)在句子里
        if re.search(r'\b' + re.escape(word) + r'\b', sent, re.IGNORECASE):
            return sent.strip()[:300] # 限制长度防止太长
    return "未找到原句"

def process_text_lite(text, vocab_df, user_limit):
    # 1. 清洗与提取
    text_lower = text.lower()
    
    # 正则提取所有单词 (长度>2, 纯字母)
    words = re.findall(r'\b[a-z]{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    found_items = []
    
    # 2. 建立高速查询字典
    # 确保列名小写
    vocab_df.columns = [c.lower() for c in vocab_df.columns]
    # word -> rank 字典
    if 'word' in vocab_df.columns and 'rank' in vocab_df.columns:
        # 转为字典加速
        vocab_dict = pd.Series(vocab_df['rank'].values, index=vocab_df['word'].astype(str)).to_dict()
    else:
        st.error("❌ 词库必须包含 'word' 和 'rank' 两列！")
        return pd.DataFrame(), pd.DataFrame()

    # 3. 匹配与分级
    for w in unique_words:
        rank = 999999
        match_word = w
        
        # 精确匹配
        if w in vocab_dict:
            rank = vocab_dict[w]
        # 简单还原规则 (去s, ed, ing)
        elif w.endswith('s') and w[:-1] in vocab_dict:
            match_word = w[:-1]
            rank = vocab_dict[match_word]
        elif w.endswith('ed') and w[:-2] in vocab_dict:
            match_word = w[:-2]
            rank = vocab_dict[match_word]
        elif w.endswith('ing') and w[:-3] in vocab_dict:
            match_word = w[:-3]
            rank = vocab_dict[match_word]
            
        # 4. 提取原句 (针对生词)
        context = ""
        is_unknown = rank > user_limit
        if is_unknown:
            context = get_sentence_context(text, w)
            
        found_items.append({
            'word': match_word,
            'rank': rank,
            'is_known': not is_unknown,
            'context': context
        })

    # 转 DataFrame
    df = pd.DataFrame(found_items)
    if not df.empty:
        known = df[df['is_known']].sort_values('rank')
        unknown = df[~df['is_known']].sort_values('rank')
        return unknown, known
    return pd.DataFrame(), pd.DataFrame()


# --- 界面 UI ---
st.title("⚡ Vibe Vocab Studio (轻量版)")
st.markdown("### 极速词汇分析 & Anki 制卡器")

# 侧边栏
st.sidebar.header("🛠️ 设置")
vocab_file = st.sidebar.file_uploader("1. 上传词频表 (CSV/Excel)", type=['csv', 'xlsx'])
user_vocab = st.sidebar.slider("2. 词汇量阈值", 1000, 20000, 6000, 500)

# 输入区
with st.expander("📝 文本输入 (支持长文本)", expanded=True):
    tab1, tab2 = st.tabs(["粘贴文本", "上传文件"])
    with tab1:
        text_input_raw = st.text_area("在此粘贴:", height=150)
    with tab2:
        uploaded_txt = st.file_uploader("上传 .txt 小说/文章", type="txt")
        if uploaded_txt:
            text_input_raw = uploaded_txt.read().decode("utf-8")

# 分析逻辑
final_text = text_input_raw if text_input_raw else ""

if st.button("🚀 开始分析", type="primary"):
    if not vocab_file:
        st.error("请先在左侧上传词库文件！")
    elif not final_text.strip():
        st.warning("请输入文本内容！")
    else:
        try:
            # 读取词库
            if vocab_file.name.endswith('.csv'):
                vocab_df = pd.read_csv(vocab_file)
            else:
                vocab_df = pd.read_excel(vocab_file)
            
            with st.spinner("正在极速分析..."):
                unknown_df, known_df = process_text_lite(final_text, vocab_df, user_vocab)
            
            st.success(f"分析完成！发现 {len(unknown_df)} 个生词。")
            
            # --- 结果展示 ---
            res_tab1, res_tab2, res_tab3 = st.tabs(["🔴 生词表", "🟢 熟词表", "🎴 Anki 制卡"])
            
            with res_tab1:
                st.dataframe(unknown_df[['word', 'rank', 'context']], use_container_width=True)
                # 简单导出
                csv = unknown_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 下载生词表 (CSV)", csv, "unknown.csv", "text/csv")

            with res_tab2:
                st.dataframe(known_df[['word', 'rank']], use_container_width=True)

            with res_tab3:
                st.markdown("#### Anki 导入文件生成")
                st.info("已自动为你提取了生词所在的【原句】。")
                
                # Anki 导出逻辑
                anki_df = pd.DataFrame()
                anki_df['Front'] = unknown_df['word']
                # 背面：原句 + <br> + 排名
                anki_df['Back'] = unknown_df['context'] + "<br><br>Rank: #" + unknown_df['rank'].astype(str)
                
                st.write("预览 (前5条):")
                st.table(anki_df.head())
                
                anki_csv = anki_df.to_csv(index=False, header=False).encode('utf-8')
                st.download_button(
                    "⚡ 下载 Anki 导入包 (.csv)", 
                    anki_csv, 
                    "anki_import.csv", 
                    "text/csv",
                    help="直接导入 Anki 即可，正面是单词，背面是原句和排名"
                )

        except Exception as e:
            st.error(f"发生错误: {e}")
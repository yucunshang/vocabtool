import streamlit as st
import pandas as pd
import re
import spacy
from collections import Counter
import io

# --- 页面配置 ---
st.set_page_config(
    page_title="COCA 词汇分级分析器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 缓存加载 NLP 模型 (解决加载慢的问题) ---
@st.cache_resource
def load_nlp():
    # 使用小模型进行词形还原 (run, running, ran -> run)
    try:
        return spacy.load("en_core_web_sm")
    except:
        # 如果云端没有模型，自动下载的命令在 packages.txt 里处理，或者这里报错提示
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

# --- 核心逻辑 ---
def process_text(text, vocab_df, user_vocab_limit):
    # 1. 文本清洗与词形还原 (Word Family)
    doc = nlp(text)
    
    # 提取单词原型 (Lemma)，过滤标点、数字、停用词
    # 这里的逻辑：只保留纯字母单词，且长度>1
    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and len(token.text) > 1]
    
    # 统计词频 (本文中的频率)
    text_word_counts = Counter(lemmas)
    
    # 2. 与 COCA 数据库比对
    # 创建一个 DataFrame
    df_text = pd.DataFrame(text_word_counts.items(), columns=['word', 'text_freq'])
    
    # 假设 vocab_df 有 'word' 和 'rank' 两列
    # 合并数据：把 COCA 的排名信息合并进来
    merged_df = pd.merge(df_text, vocab_df, on='word', how='left')
    
    # 处理未登录词 (COCA里没有的词，设为超纲)
    merged_df['rank'] = merged_df['rank'].fillna(999999).astype(int)
    
    # 3. 核心分类逻辑
    # 掌握词汇 (Within Vocabulary): 排名 <= 用户词汇量
    known_df = merged_df[merged_df['rank'] <= user_vocab_limit].copy()
    
    # 生词 (Beyond Vocabulary): 排名 > 用户词汇量
    unknown_df = merged_df[merged_df['rank'] > user_vocab_limit].copy()
    
    return known_df, unknown_df

# --- 侧边栏：设置与数据源 ---
st.sidebar.header("🛠️ 设置面板")

# 1. 必须上传 COCA 表 (为了版权安全，由用户上传)
st.sidebar.subheader("1. 数据源 (Frequency List)")
vocab_file = st.sidebar.file_uploader("上传 COCA 20000 词表 (.csv)", type=['csv'])
st.sidebar.caption("格式要求：包含两列 `word` 和 `rank` 的 CSV 文件。")

# 2. 用户参数
st.sidebar.subheader("2. 你的词汇量 (Word Families)")
user_vocab = st.sidebar.number_input(
    "输入你的词汇量阈值:", 
    min_value=1000, 
    max_value=30000, 
    value=5000, 
    step=500,
    help="例如输入 5000，系统会将 COCA 排名 5000 后的词算作生词。"
)

sort_option = st.sidebar.radio("排序方式:", ["按 COCA 词频 (由难到易)", "按字母顺序 (A-Z)"])

# --- 主界面 ---
st.title("📚 COCA 文本词汇分级工具")
st.markdown(f"**逻辑：** 自动还原单词原型 (Word Family)，对比 **COCA 排名**，通过你的词汇量 **{user_vocab}** 进行切割。")

# 输入区域
input_method = st.radio("选择输入方式:", ["直接粘贴文本", "上传 TXT 文件"], horizontal=True)

raw_text = ""
if input_method == "直接粘贴文本":
    raw_text = st.text_area("在此粘贴文本:", height=200, placeholder="Paste your English text here...")
else:
    uploaded_txt = st.file_uploader("上传 .txt 文件", type=['txt'])
    if uploaded_txt is not None:
        raw_text = uploaded_txt.read().decode("utf-8")

# --- 开始分析 ---
if st.button("🚀 开始分析", type="primary"):
    if not vocab_file:
        st.error("请先在左侧侧边栏上传 COCA 词频数据库 (CSV)！")
    elif not raw_text.strip():
        st.warning("请输入有效的文本内容。")
    else:
        # 加载数据库
        try:
            # 读取 CSV，标准化列名
            vocab_db = pd.read_csv(vocab_file)
            # 确保列名都是小写，防止用户上传的文件列名是大写
            vocab_db.columns = [c.lower() for c in vocab_db.columns]
            
            if 'word' not in vocab_db.columns or 'rank' not in vocab_db.columns:
                st.error("CSV 文件格式错误！必须包含 'word' 和 'rank' 两列。")
            else:
                vocab_db['word'] = vocab_db['word'].astype(str).str.lower().str.strip()
                
                with st.spinner("正在进行 NLP 分析与分级..."):
                    known, unknown = process_text(raw_text, vocab_db, user_vocab)
                
                # --- 排序逻辑 ---
                if sort_option == "按 COCA 词频 (由难到易)":
                    # 难词排前面 (rank 越大越难) -> 降序
                    unknown = unknown.sort_values(by='rank', ascending=False)
                    known = known.sort_values(by='rank', ascending=False)
                else:
                    unknown = unknown.sort_values(by='word', ascending=True)
                    known = known.sort_values(by='word', ascending=True)

                # --- 结果展示 ---
                st.success("分析完成！")
                
                tab1, tab2 = st.tabs([f"🔴 生词表 / 词汇量外 ({len(unknown)})", f"🟢 熟词表 / 词汇量内 ({len(known)})"])
                
                with tab1:
                    st.dataframe(
                        unknown[['word', 'rank', 'text_freq']], 
                        column_config={
                            "word": "单词 (原型)",
                            "rank": "COCA 排名",
                            "text_freq": "文中出现次数"
                        },
                        use_container_width=True
                    )
                    # 下载按钮
                    csv_unknown = unknown.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下载生词表 (CSV)", csv_unknown, "unknown_words.csv", "text/csv")
                    
                    # 生成 TXT 格式字符串
                    txt_unknown = "\n".join([f"{row['word']} (Rank: {row['rank']})" for _, row in unknown.iterrows()])
                    st.download_button("📄 下载生词表 (TXT)", txt_unknown, "unknown_words.txt", "text/plain")

                with tab2:
                    st.dataframe(
                        known[['word', 'rank', 'text_freq']],
                        column_config={
                            "word": "单词 (原型)",
                            "rank": "COCA 排名",
                            "text_freq": "文中出现次数"
                        },
                        use_container_width=True
                    )
                    csv_known = known.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下载熟词表 (CSV)", csv_known, "known_words.csv", "text/csv")

        except Exception as e:
            st.error(f"处理过程中发生错误: {e}")

# --- 页脚 ---
st.markdown("---")
st.markdown("Powered by **Streamlit** | NLP by **Spacy**")
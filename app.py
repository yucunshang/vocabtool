import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Vibe Vocab Lite", page_icon="⚡", layout="wide")

# --- 极速处理核心 (No AI, Pure Math) ---
def simple_process(text, vocab_df, user_limit):
    # 1. 简单清洗：转小写
    text = text.lower()
    
    found_items = []
    
    # 2. 单词匹配 (正则提取纯字母)
    # 正则的意思是：提取所有长度大于等于2的单词
    words = re.findall(r'\b[a-z]{2,}\b', text)
    unique_words = set(words)
    
    # 3. 建立查询索引 (加速)
    # 确保列名统一
    vocab_df.columns = [c.lower() for c in vocab_df.columns]
    # 创建字典: word -> rank
    if 'word' in vocab_df.columns and 'rank' in vocab_df.columns:
        vocab_dict = pd.Series(vocab_df['rank'].values, index=vocab_df['word'].astype(str)).to_dict()
    else:
        st.error("词库文件必须包含 'word' 和 'rank' 两列！")
        return pd.DataFrame(), pd.DataFrame()
    
    for w in unique_words:
        rank = 999999
        # 精确匹配
        if w in vocab_dict:
            rank = vocab_dict[w]
        # 简单还原 (去s, 去ed, 去ing)
        elif w.endswith('s') and w[:-1] in vocab_dict:
            rank = vocab_dict[w[:-1]]
            w = w[:-1] # 还原为单数
        elif w.endswith('ed') and w[:-2] in vocab_dict:
            rank = vocab_dict[w[:-2]]
            w = w[:-2]
        
        # 分级判断
        if rank <= user_limit:
            found_items.append({'单词': w, '类型': '熟词 (Known)', '排名': rank})
        else:
            found_items.append({'单词': w, '类型': '生词 (Unknown)', '排名': rank})

    # 转 DataFrame
    df = pd.DataFrame(found_items)
    if not df.empty:
        # 分割
        known = df[df['类型'] == '熟词 (Known)'].sort_values('排名')
        unknown = df[df['类型'] == '生词 (Unknown)'].sort_values('排名')
        return unknown, known
    return pd.DataFrame(), pd.DataFrame()

# --- 界面 ---
st.title("⚡ Vibe Vocab (极速版)")
st.caption("轻量级词汇分级工具 - 秒级响应")

# 侧边栏
st.sidebar.header("配置")
vocab_file = st.sidebar.file_uploader("1. 上传词频表 (CSV/Excel)", type=['csv', 'xlsx'])
user_vocab = st.sidebar.slider("2. 词汇量阈值", 1000, 20000, 5000, 500)

# 主界面
text_input = st.text_area("在此粘贴文本:", height=200)

if st.button("🚀 开始分析", type="primary"):
    if not vocab_file:
        st.error("请先在左侧上传词频表！")
    elif not text_input.strip():
        st.warning("请输入文本！")
    else:
        # 读取文件
        try:
            if vocab_file.name.endswith('.csv'):
                vocab_df = pd.read_csv(vocab_file)
            else:
                vocab_df = pd.read_excel(vocab_file)
            
            unknown_df, known_df = simple_process(text_input, vocab_df, user_vocab)
            
            st.success(f"分析完成！发现 {len(unknown_df)} 个生词。")
            
            tab1, tab2 = st.tabs(["🔴 生词表", "🟢 熟词表"])
            with tab1:
                st.dataframe(unknown_df, use_container_width=True)
            with tab2:
                st.dataframe(known_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"发生错误: {e}")
import streamlit as st
import pandas as pd
import re
import time
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer

# =====================================
# 页面配置
# =====================================
st.set_page_config(
    page_title="Vocab Master - Pro",
    layout="wide",
    page_icon="🚀"
)

st.title("🚀 Vocab Master - 智能 NLP 引擎")

# =====================================
# 1. 初始化 NLTK 资源 (缓存优化)
# =====================================
@st.cache_resource
def setup_nltk():
    """
    自动下载并缓存必要的 NLTK 数据包。
    """
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    return WordNetLemmatizer()

lemmatizer = setup_nltk()

# =====================================
# 2. 硬编码停用词表 (去噪关键)
# =====================================
STOPWORDS = {
    'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'was', 
    'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 
    'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 
    'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'use', 'an', 
    'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 
    'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 
    'would', 'make', 'like', 'him', 'into', 'time', 'has', 'look', 'two', 
    'more', 'write', 'go', 'see', 'number', 'no', 'way', 'could', 'people', 
    'my', 'than', 'first', 'water', 'been', 'call', 'who', 'oil', 'its', 'now', 
    'find'
}

# =====================================
# 3. 核心修改：加载 CSV 词库
# =====================================
@st.cache_data
def load_vocab():
    """
    读取同目录下的 coca_cleaned.csv
    期望格式：包含 'word' 和 'rank' 两列
    """
    csv_path = "coca_cleaned.csv"
    
    try:
        # 读取 CSV
        df = pd.read_csv(csv_path)
        
        # 1. 标准化列名：转小写并去空格 (防止 ' Rank' 这种情况)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 2. 检查关键列是否存在
        # 假设你的 CSV 列名是 'word' 和 'rank'
        # 如果你的列名是 'lemma', 'id' 等，请修改下面的字符串
        word_col = 'word'
        rank_col = 'rank'
        
        if word_col not in df.columns or rank_col not in df.columns:
            st.error(f"CSV 格式错误：未找到 '{word_col}' 或 '{rank_col}' 列。现有列名: {list(df.columns)}")
            return {}

        # 3. 数据清洗：确保 word 列是字符串，并转小写
        df[word_col] = df[word_col].astype(str).str.lower()
        
        # 4. 转换为字典 {word: rank}，查找速度 O(1)
        return dict(zip(df[word_col], df[rank_col]))
        
    except FileNotFoundError:
        st.error(f"未找到文件：{csv_path}。请确保已上传该文件。")
        return {}
    except Exception as e:
        st.error(f"读取词库失败: {e}")
        return {}

vocab_dict = load_vocab()

# =====================================
# 4. NLP 逻辑函数
# =====================================

def is_valid_word(w: str) -> bool:
    """过滤掉过短的单词、乱码或纯数字"""
    if len(w) < 2 and w not in ("a", "i"):
        return False
    if w.count("'") > 1:
        return False
    if w.isdigit():
        return False
    return True

def get_lemma(word: str) -> str:
    """
    词形还原：Running -> run
    """
    w = word.lower()
    return lemmatizer.lemmatize(w, pos='v')

def stream_analyze_text(text):
    """
    双轨分析：
    lemma_tokens -> 用于查词频表 (匹配 CSV 中的 word)
    raw_tokens   -> 用于短语识别 (保留 'United States' 原貌)
    """
    freq = Counter()
    lemma_tokens = [] 
    raw_tokens = []   

    pattern = re.compile(r"[a-zA-Z']+")

    for match in pattern.finditer(text):
        original_word = match.group()
        
        # 还原词形 (用于和 CSV 匹配)
        lemma = get_lemma(original_word)
        
        if not is_valid_word(lemma):
            continue

        freq[lemma] += 1
        lemma_tokens.append(lemma)
        raw_tokens.append(original_word.lower())

    return lemma_tokens, raw_tokens, freq

def detect_phrases(raw_tokens, min_freq=2):
    """智能短语检测 + 停用词过滤"""
    if not raw_tokens:
        return []

    bigrams = zip(raw_tokens, raw_tokens[1:])
    trigrams = zip(raw_tokens, raw_tokens[1:], raw_tokens[2:])

    phrase_cnt = Counter()

    # 过滤规则：短语首尾不能是停用词
    for bg in bigrams:
        if bg[0] not in STOPWORDS and bg[-1] not in STOPWORDS:
            phrase_cnt[" ".join(bg)] += 1
            
    for tg in trigrams:
        if tg[0] not in STOPWORDS and tg[-1] not in STOPWORDS:
            phrase_cnt[" ".join(tg)] += 1

    # 格式化输出
    results = []
    for p, f in phrase_cnt.items():
        if f >= min_freq:
            results.append((p, f))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def analyze_words(freq_dict):
    """
    核心逻辑：结合 CSV 词库计算排名
    """
    results = []
    
    # 如果词库加载失败，给出默认值防止报错
    safe_vocab = vocab_dict if vocab_dict else {}
    
    for w, f in freq_dict.items():
        if w in STOPWORDS:
            continue
            
        # 从 CSV 字典中获取排名，找不到则设为 99999 (生僻词)
        rank = safe_vocab.get(w, 99999)
        
        # 简单的难度分级逻辑（可选）
        tag = "🟢 基础"
        if rank > 5000: tag = "🟡 进阶" 
        if rank > 15000: tag = "🔴 高难/生僻"
        if rank == 99999: tag = "⚪ 未收录"

        results.append({
            "word": w,
            "rank": rank,
            "freq": f,
            "tag": tag
        })
        
    # 按排名排序 (越小越重要)，其次按频率
    results.sort(key=lambda x: (x["rank"], -x["freq"]))
    return results

# =====================================
# UI 逻辑
# =====================================

text_input = st.text_area(
    "📥 粘贴文本",
    height=220,
    placeholder="在此粘贴您的英文文章..."
)

col1, col2 = st.columns(2)
with col1:
    min_phrase_freq = st.slider("短语识别最低频率", 2, 10, 2)

if st.button("🚀 开始分析", type="primary"):

    if not text_input.strip():
        st.warning("请输入文本")
        st.stop()
        
    # 检查词库状态
    if not vocab_dict:
        st.warning("⚠️ 警告：词库文件加载失败或为空，排名功能将不可用。")

    start = time.time()

    with st.spinner("正在解析文本 & 匹配词库..."):
        
        lemma_tokens, raw_tokens, freq_dict = stream_analyze_text(text_input)
        phrases = detect_phrases(raw_tokens, min_phrase_freq)
        results = analyze_words(freq_dict)

    duration = time.time() - start

    # =================================
    # 结果展示
    # =================================
    st.success(f"✅ 分析完成，耗时 {duration:.3f} 秒")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("总词数 (Tokens)", len(raw_tokens))
    m2.metric("核心词汇量", len(results))
    m3.metric("词库覆盖率", f"{len([r for r in results if r['rank'] < 99999]) / len(results) * 100:.1f}%" if results else "0%")

    st.divider()

    left_col, right_col = st.columns([1.3, 0.7])

    with left_col:
        st.subheader("📊 词汇分级统计")
        if results:
            st.dataframe(
                results, 
                column_config={
                    "word": "单词",
                    "rank": "COCA排名",
                    "freq": "本文频次",
                    "tag": "难度分级"
                },
                use_container_width=True,
                height=600
            )
        else:
            st.info("没有发现有效单词")

    with right_col:
        st.subheader("🔗 智能短语")
        if phrases:
            st.dataframe(
                [{"Phrase": p, "Freq": f} for p, f in phrases], 
                use_container_width=True,
                height=600
            )
        else:
            st.info("未检测到高频短语")
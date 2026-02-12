import streamlit as st
import pandas as pd
import os
import sys

# ==========================================
# 0. 页面配置 (必须是第一个 Streamlit 命令)
# ==========================================
st.set_page_config(
    page_title="Vocab Master (Debug Mode)", 
    page_icon="🛠️", 
    layout="centered"
)

# ==========================================
# 1. 依赖库检查与导入
# ==========================================
# 检查 NLTK
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
except ImportError:
    st.error("❌ 缺少 nltk 库。请运行: pip install nltk")
    st.stop()

# 检查 Lemminflect (可选，没有就降级)
try:
    import lemminflect
    HAS_LEMMINFLECT = True
except ImportError:
    HAS_LEMMINFLECT = False
    st.warning("⚠️ 未检测到 lemminflect 库，将使用基础还原模式。建议运行: pip install lemminflect")

# ==========================================
# 2. 资源初始化 (带错误捕获)
# ==========================================
@st.cache_resource
def init_nlp_resources():
    status_text = st.empty()
    status_text.text("正在初始化 NLP 资源...")
    
    # 1. 下载 NLTK 数据
    nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
    nltk_path = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_path, exist_ok=True)
    nltk.data.path.append(nltk_path)
    
    for pkg in nltk_packages:
        try:
            # 先尝试查找
            nltk.data.find(f'tokenizers/{pkg}') if pkg == 'punkt' else \
            nltk.data.find(f'taggers/{pkg}') if pkg == 'averaged_perceptron_tagger' else \
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            try:
                # 找不到则下载
                nltk.download(pkg, download_dir=nltk_path, quiet=True)
            except Exception as e:
                st.error(f"❌ NLTK 数据 '{pkg}' 下载失败: {e}")
                st.info("💡 提示：如果是网络问题，请尝试挂梯子或手动下载 NLTK data。")
                
    status_text.empty()
    return True

init_nlp_resources()

# ==========================================
# 3. 数据加载 (带路径调试)
# ==========================================
@st.cache_data
def load_data():
    # 打印当前路径，帮助调试
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir) if os.path.exists(current_dir) else []
    
    # 自动寻找 csv
    target_file = "coca_cleaned.csv"
    if target_file not in files_in_dir:
        st.error(f"❌ 找不到文件: {target_file}")
        st.code(f"当前运行目录: {current_dir}\n目录下的文件: {files_in_dir}")
        return None, None
        
    try:
        df = pd.read_csv(target_file)
        # 清洗列名
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 寻找 word 和 rank 列
        w_col = next((c for c in df.columns if 'word' in c), None)
        r_col = next((c for c in df.columns if 'rank' in c), None)
        
        if not w_col or not r_col:
            st.error(f"❌ CSV 格式错误。找不到 'word' 或 'rank' 列。\n检测到的列名: {df.columns.tolist()}")
            return None, None
            
        df = df.dropna(subset=[w_col, r_col])
        df[w_col] = df[w_col].astype(str).str.lower().str.strip()
        df[r_col] = pd.to_numeric(df[r_col], errors='coerce')
        
        # 字典化
        df = df.sort_values(r_col).drop_duplicates(subset=[w_col])
        vocab_dict = pd.Series(df[r_col].values, index=df[w_col]).to_dict()
        
        return vocab_dict, df
        
    except Exception as e:
        st.error(f"❌ 读取 CSV 出错: {e}")
        return None, None

VOCAB_DICT, FULL_DF = load_data()

# ==========================================
# 4. 核心逻辑 (混合模式)
# ==========================================
def get_lemma(word, tag):
    """兼容两种库的还原逻辑"""
    if not word.isalpha(): return word
    
    # 转换 tag
    pos = 'n'
    if tag.startswith('V'): pos = 'v'
    elif tag.startswith('J'): pos = 'a'
    elif tag.startswith('R'): pos = 'r'

    # 优先使用 Lemminflect
    if HAS_LEMMINFLECT:
        try:
            upos = 'VERB' if pos == 'v' else 'ADJ' if pos == 'a' else 'ADV' if pos == 'r' else 'NOUN'
            return lemminflect.getLemma(word, upos=upos)[0]
        except:
            pass
            
    # 降级使用 WordNet
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos)

def analyze_text(text, current_lvl, target_lvl):
    if not VOCAB_DICT: return pd.DataFrame()
    
    # 简单分词 (不依赖 punkt 防止报错)
    try:
        tokens = nltk.word_tokenize(text.lower())
    except:
        import re
        tokens = re.findall(r"[a-z]+", text.lower())
        
    # 词性标注
    try:
        tagged = nltk.pos_tag(tokens)
    except:
        tagged = [(t, 'n') for t in tokens] # 失败则全默认为名词
        
    res = []
    seen = set()
    
    for word, tag in tagged:
        if len(word) < 2: continue
        lemma = get_lemma(word, tag)
        
        if lemma in seen: continue
        seen.add(lemma)
        
        rank = VOCAB_DICT.get(lemma, 99999)
        
        cat = "Beyond"
        if rank <= current_lvl: cat = "Mastered"
        elif rank <= target_lvl: cat = "Target"
        
        res.append({"Word": lemma, "Rank": rank, "Category": cat})
        
    return pd.DataFrame(res)

# ==========================================
# 5. 界面
# ==========================================
st.title("⚡️ Vocab Master (修复版)")

if FULL_DF is None:
    st.warning("⚠️ 请先解决上述报错 (缺少文件或CSV格式不对)")
else:
    txt = st.text_area("输入英文文本", height=150)
    
    if st.button("分析"):
        if not txt.strip():
            st.warning("请输入内容")
        else:
            with st.spinner("分析中..."):
                df = analyze_text(txt, 4000, 8000)
                
            if df.empty:
                st.info("未提取到单词 (或所有单词均不在词库中)")
            else:
                target_words = df[df['Category'] == 'Target'].sort_values('Rank')
                st.success(f"分析完成! 发现 {len(target_words)} 个重点生词")
                st.dataframe(target_words)
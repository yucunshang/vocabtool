import streamlit as st
import pandas as pd
import re
import os
import simplemma

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🐞", layout="wide")

# ==========================================
# 🐞 v6.0 核心修复与诊断模块
# ==========================================

# 1. 强力 Simplemma 加载 (带自我检测)
LEMMA_STATUS = "未知"
try:
    # 尝试新版 (v1.0+)
    test = simplemma.lemmatize("went", lang="en")
    def get_lemma(word): return simplemma.lemmatize(word, lang="en")
    if test == "go": 
        LEMMA_STATUS = "✅ 正常 (v1.x)"
    else:
        LEMMA_STATUS = f"⚠️ 异常 (返回: {test})"
except TypeError:
    # 尝试旧版
    try:
        lang_data = simplemma.load_data('en')
        def get_lemma(word): return simplemma.lemmatize(word, lang_data)
        if get_lemma("went") == "go":
            LEMMA_STATUS = "✅ 正常 (v0.9)"
        else:
            LEMMA_STATUS = "⚠️ 异常 (旧版加载失败)"
    except:
        def get_lemma(word): return word
        LEMMA_STATUS = "❌ 失败 (无法加载库)"

# 2. 强力词库加载 (指定列名读取)
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]

@st.cache_data
def load_vocab_debug():
    file_path = None
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            file_path = f
            break
    
    if not file_path: return None, "未找到任何 csv 文件", {}

    try:
        df = None
        # 专门针对 coca_cleaned.csv 的优化读取
        if "cleaned" in file_path:
            # 既然是 cleaned，我们假定它没有表头，或者表头是标准英文
            # 尝试直接指定列名读取，强制修复
            try:
                # 尝试当作无表头读取
                df_test = pd.read_csv(file_path, header=None)
                # 检查第一行是不是 word, rank
                first_cell = str(df_test.iloc[0,0])
                if 'word' in first_cell.lower():
                     # 有表头
                     df = pd.read_csv(file_path)
                else:
                     # 无表头，手动指定
                     df = pd.read_csv(file_path, header=None, names=['word', 'rank'])
            except:
                pass

        # 如果上面的专用读取没跑通，走通用读取
        if df is None:
            for enc in ['utf-8-sig', 'utf-8', 'gbk']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    if len(df) > 10: break
                except: continue

        if df is None: return None, "文件读取失败 (编码错误?)", {}

        # 统一列名
        df.columns = [str(c).strip().lower() for c in df.columns]
        cols = list(df.columns)

        # 寻找 word 和 rank
        w_col, r_col = None, None
        
        # 策略1: 精确匹配
        if 'word' in cols and 'rank' in cols:
            w_col, r_col = 'word', 'rank'
        # 策略2: 位置猜测 (针对 coca_cleaned)
        elif len(cols) == 2:
            w_col, r_col = df.columns[0], df.columns[1]
        # 策略3: 原始文件猜测
        elif len(cols) >= 4:
            w_col, r_col = df.columns[0], df.columns[3]
        
        # 策略4: 关键词搜索
        if not w_col: w_col = next((c for c in df.columns if 'word' in c or '单词' in c), None)
        if not r_col: r_col = next((c for c in df.columns if 'rank' in c or '排序' in c or '词频' in c), None)

        if not w_col or not r_col:
            return df, f"列名识别失败: {cols}", {}

        # 提取数据
        df['word_clean'] = df[w_col].astype(str).str.lower().str.strip()
        df['rank_clean'] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
        
        # 过滤
        df = df[df['rank_clean'] > 0]
        df = df[df['rank_clean'] < 99999]
        
        vocab_dict = pd.Series(df['rank_clean'].values, index=df['word_clean']).to_dict()
        
        # 诊断信息
        debug_info = {
            "file": file_path,
            "cols": cols,
            "used_cols": (w_col, r_col),
            "sample_the": vocab_dict.get('the', '未找到'),
            "sample_good": vocab_dict.get('good', '未找到'),
            "count": len(vocab_dict)
        }
        
        return df, "成功", vocab_dict

    except Exception as e:
        return None, str(e), {}

# 加载数据
df_raw, status, vocab_dict = load_vocab_debug()

# ==========================================
# 🔍 侧边栏：诊断面板 (Debug Panel)
# ==========================================
st.sidebar.title("🛠️ 诊断面板")
st.sidebar.info("如果结果不对，请截图这里发给我！")

with st.sidebar.expander("1. 还原引擎检测", expanded=True):
    st.write(f"状态: {LEMMA_STATUS}")
    st.caption("测试: went -> " + get_lemma("went"))

with st.sidebar.expander("2. 词库读取检测", expanded=True):
    if vocab_dict:
        debug = load_vocab_debug()[2] # 重新获取debug info
        st.write(f"📂 文件: `{debug['file']}`")
        st.write(f"🔢 总词数: `{debug['count']}`")
        
        st.markdown("---")
        st.write("**关键词排名检查:**")
        
        # 检查 'the'
        rank_the = debug['sample_the']
        icon_the = "✅" if rank_the == 1 else "❌"
        st.write(f"🔹 'the': {rank_the} {icon_the}")
        
        # 检查 'good'
        rank_good = debug['sample_good']
        st.write(f"🔹 'good': {rank_good}")
        
        if rank_the == '未找到' or rank_the > 100:
            st.error("🚨 严重错误：基础词排名不对！一定是列读取错了。")
    else:
        st.error(f"加载失败: {status}")

# ==========================================
# 主程序逻辑
# ==========================================

st.sidebar.divider()
st.sidebar.subheader("学习设置")
vocab_range = st.sidebar.slider("选择区间", 1, 20000, (6000, 8000), 500)
range_start, range_end = vocab_range

st.title("🐞 Vibe Vocab v6.0 (诊断版)")

if not vocab_dict:
    st.warning("⚠️ 请先解决左侧的报错")
    st.stop()

def process_text_debug(text, vocab_dict, r_start, r_end):
    text_lower = text.lower()
    words = re.findall(r'\b[a-z\']{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    known, target, beyond = [], [], []
    
    for w in unique_words:
        rank = 99999
        match = w
        
        # 1. 查原形
        if w in vocab_dict:
            rank = vocab_dict[w]
        else:
            # 2. 查还原
            lemma = get_lemma(w)
            if lemma in vocab_dict:
                rank = vocab_dict[lemma]
                match = lemma
            else:
                # 3. 查变体
                if w.endswith("'s") and w[:-2] in vocab_dict:
                    rank = vocab_dict[w[:-2]]
                    match = w[:-2]

        item = {'单词': match, '原文': w, '排名': int(rank)}
        
        if rank <= r_start: known.append(item)
        elif r_start < rank <= r_end: target.append(item)
        else:
            if match == w: item['原文'] = '-'
            beyond.append(item)
            
    return pd.DataFrame(known), pd.DataFrame(target), pd.DataFrame(beyond)

# 输入区
text_input = st.text_area("在此粘贴文本:", height=150)

if st.button("🚀 开始分析"):
    if not text_input.strip():
        st.warning("请输入文本")
    else:
        df_k, df_t, df_b = process_text_debug(text_input, vocab_dict, range_start, range_end)
        
        st.success("分析完成")
        t1, t2, t3 = st.tabs([
            f"🟡 重点 ({len(df_t)})", 
            f"🔴 超纲 ({len(df_b)})", 
            f"🟢 熟词 ({len(df_k)})"
        ])
        
        with t1: st.dataframe(df_t, use_container_width=True)
        with t2: st.dataframe(df_b, use_container_width=True)
        with t3: st.dataframe(df_k, use_container_width=True)
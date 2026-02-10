import streamlit as st
import pandas as pd
import re
import os
import simplemma

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🛡️", layout="wide")

# ==========================================
# 1. 兼容性最强的 Lemmatizer (还原引擎)
# ==========================================
# 不再检测版本，直接定义一个能容错的函数
try:
    # 尝试加载数据 (旧版逻辑)
    if hasattr(simplemma, 'load_data'):
        LANG_DATA = simplemma.load_data('en')
    else:
        LANG_DATA = None
except:
    LANG_DATA = None

def get_lemma(word):
    """最稳健的还原函数"""
    try:
        # 优先尝试新版 (v1.x) 直接调用
        return simplemma.lemmatize(word, lang='en')
    except TypeError:
        # 如果报错，说明是旧版，需要传数据
        if LANG_DATA:
            return simplemma.lemmatize(word, LANG_DATA)
        return word # 实在不行返回原词
    except Exception:
        return word

# ==========================================
# 2. 带“自检”功能的词库加载 (核心修复)
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]

@st.cache_data
def load_vocab_robust():
    # 1. 找到文件
    file_path = None
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            file_path = f
            break
    
    if not file_path:
        return None, "❌ 未找到 csv 文件，请确保文件已上传到 GitHub。"

    # 2. 尝试多种方式读取，直到通过“自检”
    success_dict = None
    debug_msg = []

    # 定义读取策略
    strategies = [
        # 策略A: 标准CSV (有表头)
        {'args': {'encoding': 'utf-8-sig'}, 'desc': 'UTF-8 标准读取'},
        {'args': {'encoding': 'utf-8'}, 'desc': 'UTF-8 读取'},
        {'args': {'encoding': 'gbk'}, 'desc': 'GBK 读取'},
        # 策略B: 无表头 (假设第一列单词，第二列排名)
        {'args': {'encoding': 'utf-8', 'header': None}, 'desc': '无表头模式'},
    ]

    for strat in strategies:
        try:
            df = pd.read_csv(file_path, **strat['args'])
            
            # 统一列名 (如果是无表头模式，手动指定)
            if strat.get('args', {}).get('header') is None:
                # 假设前两列有效
                if len(df.columns) >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ['word', 'rank']
            else:
                # 标准化列名
                df.columns = [str(c).strip().lower() for c in df.columns]

            # 寻找 word 和 rank 列
            w_col = next((c for c in df.columns if any(k in c for k in ['word', '单词', '词汇'])), None)
            r_col = next((c for c in df.columns if any(k in c for k in ['rank', '排名', '序号', '词频'])), None)

            # 兜底列名 (如果没有匹配到，且只有2列，就盲猜)
            if not w_col and len(df.columns) == 2: w_col = df.columns[0]
            if not r_col and len(df.columns) == 2: r_col = df.columns[1]

            # 也是兜底 (针对原始乱文件)
            if not w_col and len(df.columns) >= 4: w_col = df.columns[0]
            if not r_col and len(df.columns) >= 4: r_col = df.columns[3]

            if not w_col or not r_col:
                continue # 列都没找齐，换下一种策略

            # 清洗数据
            df['w_clean'] = df[w_col].astype(str).str.lower().str.strip()
            df['r_clean'] = pd.to_numeric(df[r_col], errors='coerce')
            
            # 去除无效行
            df_valid = df.dropna(subset=['r_clean'])
            
            # 生成字典
            temp_dict = pd.Series(df_valid['r_clean'].values, index=df_valid['w_clean']).to_dict()

            # === 关键步骤：自我核验 (Sanity Check) ===
            # 检查基础词 'the', 'of', 'and' 的排名是否合理
            # 它们应该是前 10 名
            score = 0
            if temp_dict.get('the', 999) <= 10: score += 1
            if temp_dict.get('of', 999) <= 10: score += 1
            if temp_dict.get('and', 999) <= 10: score += 1

            if score >= 1:
                # 通过核验！
                success_dict = temp_dict
                debug_msg.append(f"✅ 策略 [{strat['desc']}] 成功! 'the' rank: {temp_dict.get('the')}")
                break
            else:
                debug_msg.append(f"⚠️ 策略 [{strat['desc']}] 失败: 'the' rank is {temp_dict.get('the')}")

        except Exception as e:
            debug_msg.append(f"❌ 策略 [{strat['desc']}] 报错: {str(e)}")
            continue

    if success_dict:
        return success_dict, f"加载成功 ({len(success_dict)}词)"
    else:
        return None, f"所有读取策略都失败。\n调试日志:\n" + "\n".join(debug_msg)


# ==========================================
# 3. 核心逻辑
# ==========================================
vocab_dict, status_msg = load_vocab_robust()

st.title("🛡️ Vibe Vocab v7.0 (最终核验版)")

# 侧边栏状态
if vocab_dict:
    st.sidebar.success(status_msg)
    # 双重保险显示
    the_rank = vocab_dict.get('the', 'Not Found')
    st.sidebar.info(f"检查点: 'the' = {the_rank}")
else:
    st.error("💥 严重错误：词库加载失败")
    st.text(status_msg)
    st.stop()

st.sidebar.divider()
vocab_range = st.sidebar.slider("设定学习范围", 1, 20000, (6000, 8000), 500)
r_start, r_end = vocab_range

st.sidebar.markdown(f"""
- 🟢 **熟词**: 1 ~ {r_start}
- 🟡 **重点**: {r_start} ~ {r_end}
- 🔴 **超纲**: {r_end}+
""")

# 处理逻辑
def process_text(text):
    text_lower = text.lower()
    words = re.findall(r'\b[a-z\']{2,}\b', text_lower)
    unique_words = sorted(list(set(words)))
    
    known, target, beyond = [], [], []
    
    for w in unique_words:
        rank = 99999
        match = w
        note = ""

        # 1. 直接查
        if w in vocab_dict:
            rank = vocab_dict[w]
        else:
            # 2. 还原查 (went -> go)
            lemma = get_lemma(w)
            if lemma in vocab_dict:
                rank = vocab_dict[lemma]
                match = lemma
                note = f"(原: {w})"
            else:
                # 3. 简单去尾查
                if w.endswith("s") and w[:-1] in vocab_dict:
                    rank = vocab_dict[w[:-1]]
                    match = w[:-1]
                elif w.endswith("'s") and w[:-2] in vocab_dict:
                    rank = vocab_dict[w[:-2]]
                    match = w[:-2]

        item = {'单词': match, '排名': int(rank), '备注': note}
        
        if rank <= r_start: known.append(item)
        elif r_start < rank <= r_end: target.append(item)
        else: beyond.append(item)

    return pd.DataFrame(known), pd.DataFrame(target), pd.DataFrame(beyond)

# 界面
text_input = st.text_area("在此粘贴文本:", height=150)

if st.button("🚀 开始分析", type="primary"):
    if not text_input.strip():
        st.warning("请输入内容")
    else:
        df_k, df_t, df_b = process_text(text_input)
        
        st.success("分析完成")
        t1, t2, t3 = st.tabs([
            f"🟡 重点词 ({len(df_t)})", 
            f"🔴 超纲词 ({len(df_b)})", 
            f"🟢 熟词 ({len(df_k)})"
        ])
        
        def dl_btn(df, name):
            if df.empty: return
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 下载 {name}.csv", csv, f"{name}.csv", "text/csv")

        with t1:
            st.dataframe(df_t, use_container_width=True)
            dl_btn(df_t, "target_words")
            
        with t2:
            st.dataframe(df_b, use_container_width=True)
            dl_btn(df_b, "beyond_words")
            
        with t3:
            st.dataframe(df_k, use_container_width=True)
            dl_btn(df_k, "known_words")
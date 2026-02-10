import streamlit as st
import pandas as pd
import re
import os
import simplemma

st.set_page_config(page_title="Vibe Vocab Studio", page_icon="🕵️", layout="wide")

st.title("🕵️ Vibe Vocab v8.0 (透明调试版)")
st.caption("所见即所得 · 拒绝黑盒操作")

# ==========================================
# 1. 基础环境检查
# ==========================================
# 检查 simplemma 是否能工作
LEMMA_Check = "❌ 损坏"
try:
    test = simplemma.lemmatize("went", lang="en")
    if test == "go":
        LEMMA_Check = "✅ 正常 (v1.x)"
        def get_lemma(word): return simplemma.lemmatize(word, lang="en")
    else:
        # 尝试旧版
        if hasattr(simplemma, 'load_data'):
            lang_data = simplemma.load_data('en')
            def get_lemma(word): return simplemma.lemmatize(word, lang_data)
            LEMMA_Check = "✅ 正常 (v0.9)"
        else:
            LEMMA_Check = "⚠️ 异常 (返回原词)"
            def get_lemma(word): return word
except:
    LEMMA_Check = "❌ 彻底失败"
    def get_lemma(word): return word

# ==========================================
# 2. 读取文件 (只读，不猜)
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv", "COCA20000词Excel版.xlsx - Sheet1.csv"]

@st.cache_data
def load_raw_df():
    file_path = None
    for f in POSSIBLE_FILES:
        if os.path.exists(f):
            file_path = f
            break
    
    if not file_path: return None, "未找到文件"

    # 尝试暴力读取
    for enc in ['utf-8', 'utf-8-sig', 'gbk']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            if len(df) > 1:
                # 统一转成字符串列名，防止出错
                df.columns = [str(c).strip() for c in df.columns]
                return df, file_path
        except:
            continue
    return None, "读取失败"

df_raw, msg = load_raw_df()

if df_raw is None:
    st.error(f"❌ 致命错误: {msg}")
    st.stop()

# ==========================================
# 3. 交互式配置 (把控制权交给你)
# ==========================================
with st.sidebar:
    st.header("🛠️ 核心设置")
    st.info(f"词形还原引擎: {LEMMA_Check}")
    
    st.write("---")
    st.write("### 1. 确认数据列")
    st.caption(f"当前加载: {os.path.basename(msg)}")
    
    # 让用户自己选列！
    all_cols = list(df_raw.columns)
    
    # 尝试预选
    default_word = next((c for c in all_cols if 'word' in c.lower() or '单词' in c), all_cols[0])
    default_rank = next((c for c in all_cols if 'rank' in c.lower() or '排序' in c or '词频' in c), all_cols[1] if len(all_cols)>1 else all_cols[0])

    col_word = st.selectbox("哪一列是【单词】?", all_cols, index=all_cols.index(default_word))
    col_rank = st.selectbox("哪一列是【排名】?", all_cols, index=all_cols.index(default_rank))

    # 生成字典
    try:
        # 清洗
        df_raw['clean_word'] = df_raw[col_word].astype(str).str.lower().str.strip()
        df_raw['clean_rank'] = pd.to_numeric(df_raw[col_rank], errors='coerce').fillna(99999)
        
        # 建立索引
        vocab_dict = pd.Series(df_raw['clean_rank'].values, index=df_raw['clean_word']).to_dict()
        
        st.success(f"✅ 索引建立完成: {len(vocab_dict)} 词")
    except Exception as e:
        st.error(f"建立索引失败: {e}")
        st.stop()
        
    st.write("---")
    vocab_range = st.slider("学习区间", 1, 20000, (6000, 8000), 500)

# ==========================================
# 4. 数据透视区 (关键！)
# ==========================================
with st.expander("📊 查看词库前 10 行 (排错必看)", expanded=True):
    st.write("请检查：1. 列名选对了吗？ 2. 'the' 的排名是 1 吗？")
    st.dataframe(df_raw[[col_word, col_rank]].head(10), use_container_width=True)

# ==========================================
# 5. 单词侦探 (Debug 专用)
# ==========================================
st.divider()
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("🕵️ 单词侦探")
    debug_word = st.text_input("输入一个词测试 (如 went):", placeholder="试一下简单的词...")
    
    if debug_word:
        w = debug_word.lower().strip()
        lemma = get_lemma(w)
        
        st.write(f"1. 原始词: **{w}**")
        
        # 查原始
        if w in vocab_dict:
            r = vocab_dict[w]
            st.write(f"   - 在词库中? ✅ (排名: {r})")
        else:
            st.write(f"   - 在词库中? ❌")
            
        st.write(f"2. 还原词: **{lemma}**")
        
        # 查还原
        if lemma in vocab_dict:
            r = vocab_dict[lemma]
            st.write(f"   - 在词库中? ✅ (排名: {r})")
            final_rank = r
        else:
            st.write(f"   - 在词库中? ❌")
            final_rank = 99999
            
        # 判定
        limit = vocab_range[0]
        if final_rank <= limit:
            st.success(f"结论: 🟢 熟词 (排名 {final_rank} <= {limit})")
        else:
            st.error(f"结论: 🔴 生词/超纲 (排名 {final_rank} > {limit})")

# ==========================================
# 6. 批量分析逻辑
# ==========================================
with c2:
    st.subheader("📝 批量分析")
    text_input = st.text_area("输入文章:", height=150)
    
    if st.button("🚀 开始分析"):
        if not text_input: st.warning("没内容啊")
        else:
            words = re.findall(r'\b[a-z\']{2,}\b', text_input.lower())
            unique_words = sorted(list(set(words)))
            
            res = []
            for w in unique_words:
                rank = 99999
                match = w
                
                # 查词逻辑
                if w in vocab_dict:
                    rank = vocab_dict[w]
                else:
                    lemma = get_lemma(w)
                    if lemma in vocab_dict:
                        rank = vocab_dict[lemma]
                        match = lemma
                    elif w.endswith("s") and w[:-1] in vocab_dict:
                         rank = vocab_dict[w[:-1]]
                         match = w[:-1]
                
                res.append({'单词': match, '原文': w, '排名': int(rank)})
            
            df_res = pd.DataFrame(res)
            
            # 分级
            r1, r2 = vocab_range
            df_k = df_res[df_res['排名'] <= r1]
            df_t = df_res[(df_res['排名'] > r1) & (df_res['排名'] <= r2)]
            df_b = df_res[df_res['排名'] > r2]
            
            t1, t2, t3 = st.tabs([f"重点 ({len(df_t)})", f"超纲 ({len(df_b)})", f"熟词 ({len(df_k)})"])
            with t1: st.dataframe(df_t, use_container_width=True)
            with t2: st.dataframe(df_b, use_container_width=True)
            with t3: st.dataframe(df_k, use_container_width=True)
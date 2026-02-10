import streamlit as st
import pandas as pd
import os

# ==========================================
# 1. 界面配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Lookup", page_icon="🔍")

st.markdown("""
<style>
    /* 调整输入框字体，方便查看 */
    .stTextArea textarea {
        font-size: 16px !important;
        font-family: 'Consolas', 'Courier New', monospace;
        line-height: 1.6;
    }
    /* 隐藏顶部彩条和页脚 */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; }
    
    /* 结果列表样式 */
    .result-box {
        font-family: 'Consolas', monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 词库加载
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv"]

@st.cache_data
def load_vocab():
    file_path = next((f for f in POSSIBLE_FILES if os.path.exists(f)), None)
    if not file_path: return None
    
    try:
        df = pd.read_csv(file_path)
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = cols
        
        w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
        r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1])
        
        df[w_col] = df[w_col].astype(str).str.lower().str.strip()
        df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
        
        return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
    except:
        return None

vocab_dict = load_vocab()

# ==========================================
# 3. 主界面逻辑
# ==========================================

# --- 顶部设置 ---
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    current_level = st.number_input("当前水平 (Current)", 0, 20000, 9000, 500)
with c2:
    target_level = st.number_input("目标水平 (Target)", 0, 20000, 15000, 500)

st.divider()

# --- 左右分栏 ---
left, right = st.columns([1, 1])

with left:
    st.markdown("### 📝 输入单词 (空格或换行分隔)")
    text_input = st.text_area(
        "input_area", 
        height=600, 
        placeholder="marina knockout   warehouse\nbubonic trivia", 
        label_visibility="collapsed"
    )
    analyze_btn = st.button("⚡ 查询 / Lookup", type="primary", use_container_width=True)

with right:
    st.markdown("### 📊 分级结果")
    
    if not vocab_dict:
        st.error("❌ 未找到词库文件 (coca_cleaned.csv)")
    elif analyze_btn and text_input:
        
        # === 核心优化：智能分割 ===
        # text_input.split() 不带参数时，会自动处理：
        # 1. 换行符 \n
        # 2. 单个空格
        # 3. 连续多个空格
        # 4. Tab 键
        # 把它变成干净的单词列表
        words = text_input.split()
        
        data = []
        
        for word in words:
            word_clean = word.strip()
            if not word_clean: continue
            
            # 纯查表逻辑 (不还原，不修改)
            lookup_key = word_clean.lower()
            rank = vocab_dict.get(lookup_key, 99999)
            
            category = "beyond"
            if rank <= current_level:
                category = "known"
            elif rank <= target_level:
                category = "target"
            
            data.append({
                "word": word_clean,
                "rank": rank,
                "category": category
            })
            
        # 生成结果
        df = pd.DataFrame(data)
        
        if not df.empty:
            t1, t2, t3 = st.tabs([
                f"🟡 重点 ({len(df[df['category']=='target'])})", 
                f"🔴 超纲 ({len(df[df['category']=='beyond'])})", 
                f"🟢 已掌握 ({len(df[df['category']=='known'])})"
            ])
            
            def show_list(category_name):
                subset = df[df['category'] == category_name]
                if subset.empty:
                    st.info("列表为空")
                else:
                    lines_out = []
                    for _, row in subset.iterrows():
                        r_str = str(int(row['rank'])) if row['rank'] != 99999 else "-"
                        lines_out.append(f"{row['word']} ({r_str})")
                    
                    st.text_area(f"{category_name}_out", value="\n".join(lines_out), height=500, label_visibility="collapsed")

            with t1: show_list("target")
            with t2: show_list("beyond")
            with t3: show_list("known")
            
        else:
            st.warning("请输入有效内容")

    elif not text_input:
        st.info("👈 请在左侧输入单词")
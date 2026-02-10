import streamlit as st
import pandas as pd
import os

# ==========================================
# 1. 极简配置
# ==========================================
st.set_page_config(layout="wide", page_title="Direct Vocab Lookup", page_icon="🔍")

st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
        font-family: 'Consolas', 'Courier New', monospace;
        line-height: 1.5;
    }
    .stNumberInput input { font-weight: bold; color: #1a73e8; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 词库加载 (coca_cleaned.csv)
# ==========================================
POSSIBLE_FILES = ["coca_cleaned.csv", "data.csv"]

@st.cache_data
def load_vocab():
    file_path = next((f for f in POSSIBLE_FILES if os.path.exists(f)), None)
    if not file_path: return None
    
    try:
        df = pd.read_csv(file_path)
        # 极简清洗：只认 word 和 rank 列，忽略大小写
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = cols
        
        # 智能匹配列名
        w_col = next((c for c in cols if 'word' in c or '单词' in c), cols[0])
        r_col = next((c for c in cols if 'rank' in c or '排序' in c), cols[1])
        
        # 建立查词字典：key=word(lower), value=rank
        df[w_col] = df[w_col].astype(str).str.lower().str.strip()
        df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(99999)
        
        return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
    except:
        return None

vocab_dict = load_vocab()

# ==========================================
# 3. 界面布局
# ==========================================

# 顶部设置栏
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    current_level = st.number_input("当前水平 (Current)", 0, 20000, 6000, 500)
with c2:
    target_level = st.number_input("目标水平 (Target)", 0, 20000, 8000, 500)

st.divider()

# 左右分栏
left, right = st.columns([1, 1])

with left:
    st.markdown("### 📝 输入列表")
    text_input = st.text_area(
        "input_area", 
        height=600, 
        placeholder="在此粘贴单词列表（每行一个）...\nmarina\nknockout", 
        label_visibility="collapsed"
    )
    analyze_btn = st.button("⚡ 开始查询 / Lookup", type="primary", use_container_width=True)

with right:
    st.markdown("### 📊 查询结果")
    
    if not vocab_dict:
        st.error("❌ 未找到词库文件 (coca_cleaned.csv)")
    elif analyze_btn and text_input:
        
        # 1. 逐行处理输入 (不去重，不修改，只strip)
        lines = text_input.split('\n')
        
        data = []
        
        # 2. 查词逻辑 (纯粹查表)
        for line in lines:
            word_to_check = line.strip()
            if not word_to_check: continue # 跳过空行
            
            # 转小写去查 (词库Key是小写的)，但显示用原样
            lookup_key = word_to_check.lower()
            rank = vocab_dict.get(lookup_key, 99999)
            
            # 分组逻辑
            category = "beyond"
            if rank <= current_level:
                category = "known"
            elif rank <= target_level:
                category = "target"
            
            data.append({
                "word": word_to_check, # 保持原样显示
                "rank": rank,
                "category": category
            })
            
        # 3. 生成结果
        df = pd.DataFrame(data)
        
        if not df.empty:
            # 这里的排序如果你不需要也可以去掉，目前是按 Rank 排一下方便看
            # df = df.sort_values('rank') 
            
            t1, t2, t3 = st.tabs([
                f"🟡 重点 ({len(df[df['category']=='target'])})", 
                f"🔴 超纲/未收录 ({len(df[df['category']=='beyond'])})", 
                f"🟢 已掌握 ({len(df[df['category']=='known'])})"
            ])
            
            def show_list(category_name):
                subset = df[df['category'] == category_name]
                if subset.empty:
                    st.info("列表为空")
                else:
                    # 格式：word (rank)
                    # 如果rank是99999，显示 (未收录)
                    lines_out = []
                    for _, row in subset.iterrows():
                        r_display = str(row['rank']) if row['rank'] != 99999 else "-"
                        lines_out.append(f"{row['word']} ({r_display})")
                        
                    st.text_area(f"{category_name}_out", value="\n".join(lines_out), height=500, label_visibility="collapsed")

            with t1: show_list("target")
            with t2: show_list("beyond")
            with t3: show_list("known")
            
        else:
            st.warning("请输入有效文本。")

    elif not text_input:
        st.info("👈 请在左侧粘贴单词列表")
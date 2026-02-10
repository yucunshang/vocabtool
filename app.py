import streamlit as st
import pandas as pd
import re
import os
import lemminflect

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
        font-family: 'Consolas', 'Courier New', monospace;
        line-height: 1.6;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 智能还原引擎 (Lemminflect)
# ==========================================
def smart_lemmatize(text):
    """
    使用 lemminflect 进行精准还原。
    特点：保留形容词 (excited -> excited)，还原动词 (went -> go)
    """
    # 简单的分词 (保留单词和撇号)
    words = re.findall(r"[a-zA-Z']+", text)
    
    results = []
    for w in words:
        # 1. 获取所有可能的 lemma (还原形式)
        # getAllLemmas 返回一个字典 {'VERB': 'go', 'NOUN': '...'}
        lemmas_dict = lemminflect.getAllLemmas(w)
        
        # 2. 智能选择策略
        if not lemmas_dict:
            # 如果没查到，就返回原词小写
            results.append(w.lower())
            continue
            
        # 优先保留形容词 (ADJ) 和 副词 (ADV)，防止 excited -> excite
        if 'ADJ' in lemmas_dict:
            lemma = lemmas_dict['ADJ'][0]
        elif 'ADV' in lemmas_dict:
            lemma = lemmas_dict['ADV'][0]
        elif 'VERB' in lemmas_dict:
            lemma = lemmas_dict['VERB'][0]
        elif 'NOUN' in lemmas_dict:
            lemma = lemmas_dict['NOUN'][0]
        else:
            # 兜底：取第一个
            lemma = list(lemmas_dict.values())[0][0]
            
        results.append(lemma)
        
    return " ".join(results)

# ==========================================
# 3. 词库加载
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
    except: return None

vocab_dict = load_vocab()

# ==========================================
# 4. 界面布局
# ==========================================
st.title("🚀 Vocab Master Pro (Smart Lemma)")

tab_lemma, tab_grade = st.tabs(["🛠️ 1. 智能还原 (Smart Restore)", "📊 2. 单词分级 (Vocab Grader)"])

# ---------------------------------------------------------
# Tab 1: 智能还原
# ---------------------------------------------------------
with tab_lemma:
    st.caption("功能：智能还原文章单词。保留形容词状态 (excited -> excited)，还原动词时态 (went -> go)。")
    
    c1, c2 = st.columns(2)
    with c1:
        raw_text = st.text_area("输入原始文章", height=400, placeholder="He was excited about the trip.\nShe went home.")
        btn_restore = st.button("开始还原", type="primary")
        
    with c2:
        if btn_restore and raw_text:
            # 调用 lemminflect
            res = smart_lemmatize(raw_text)
            st.text_area("还原结果", value=res, height=400)
        elif not raw_text:
            st.info("👈 请输入文本")

# ---------------------------------------------------------
# Tab 2: 单词分级 (保持之前的逻辑)
# ---------------------------------------------------------
with tab_grade:
    st.caption("功能：查单词/短语排名。支持 'quantum entanglement' 这种短语。")
    
    # 顶部设置
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        current_level = st.number_input("当前水平", 0, 20000, 9000, 500)
    with col_b:
        target_level = st.number_input("目标水平", 0, 20000, 15000, 500)
    
    st.divider()
    
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("##### 输入列表")
        # 模式选择
        input_mode = st.radio(
            "识别模式:",
            ("自动分词 (Word Mode)", "按行处理 (Phrase Mode)"),
            horizontal=True,
            help="【自动分词】适合乱序单词堆。\n【按行处理】适合短语列表 (如 quantum entanglement)。"
        )
        grade_input = st.text_area("input_box", height=400, placeholder="marina\nquantum entanglement", label_visibility="collapsed")
        btn_grade = st.button("开始分级", type="primary", use_container_width=True)

    with g_col2:
        st.markdown("##### 分级结果")
        
        if not vocab_dict:
            st.error("❌ 词库未加载")
        elif btn_grade and grade_input:
            
            # 处理输入
            items_to_check = []
            if "按行处理" in input_mode:
                lines = grade_input.split('\n')
                for line in lines:
                    if line.strip(): items_to_check.append(line.strip())
            else:
                items_to_check = grade_input.split()
            
            # 查词
            data = []
            for item in items_to_check:
                lookup_key = item.lower()
                rank = vocab_dict.get(lookup_key, 99999)
                
                cat = "beyond"
                if rank <= current_level: cat = "known"
                elif rank <= target_level: cat = "target"
                
                data.append({"word": item, "rank": rank, "cat": cat})
            
            # 显示结果
            df = pd.DataFrame(data)
            if not df.empty:
                t1, t2, t3 = st.tabs([
                    f"🟡 重点 ({len(df[df['cat']=='target'])})", 
                    f"🔴 超纲 ({len(df[df['cat']=='beyond'])})", 
                    f"🟢 已掌握 ({len(df[df['cat']=='known'])})"
                ])
                
                def show(cat_name):
                    sub = df[df['cat'] == cat_name]
                    if sub.empty: st.info("无")
                    else:
                        txt = "\n".join([f"{r['word']} ({r['rank'] if r['rank']!=99999 else '-'})" for _, r in sub.iterrows()])
                        st.text_area(f"{cat_name}_res", value=txt, height=400, label_visibility="collapsed")

                with t1: show("target")
                with t2: show("beyond")
                with t3: show("known")
            else:
                st.warning("无有效输入")
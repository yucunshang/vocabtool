import streamlit as st
import pandas as pd
import re
import os
import lemminflect
import nltk

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="Vocab Master Pro", page_icon="🚀")

st.markdown("""
<style>
    .stCode {
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 16px !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 初始化 NLP 引擎 (NLTK + Lemminflect)
# ==========================================
@st.cache_resource
def download_nltk_resources():
    """下载必要的 NLTK 数据包 (只运行一次)"""
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_resources()

def get_display_case(word):
    """
    智能判断大小写：
    1. Sydney -> Sydney (专有名词保留大写)
    2. ANTI -> anti (普通词强制小写)
    3. Table -> table (普通词强制小写)
    """
    # 先把词变成 Title Case (首字母大写) 去测试，这样 NLTK 判断最准
    test_word = word.title()
    
    # 获取词性标注
    # NNP/NNPS 代表专有名词 (Proper Noun)
    tags = nltk.pos_tag([test_word])
    pos_tag = tags[0][1]
    
    if pos_tag.startswith('NNP'):
        return test_word # 是人名/地名，返回 Sydney
    else:
        return word.lower() # 是普通词，返回 anti

def smart_lemmatize(text):
    words = re.findall(r"[a-zA-Z']+", text)
    results = []
    for w in words:
        lemmas_dict = lemminflect.getAllLemmas(w)
        if not lemmas_dict:
            results.append(w.lower())
            continue
            
        if 'ADJ' in lemmas_dict: lemma = lemmas_dict['ADJ'][0]
        elif 'ADV' in lemmas_dict: lemma = lemmas_dict['ADV'][0]
        elif 'VERB' in lemmas_dict: lemma = lemmas_dict['VERB'][0]
        elif 'NOUN' in lemmas_dict: lemma = lemmas_dict['NOUN'][0]
        else: lemma = list(lemmas_dict.values())[0][0]
            
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
        
        # 排序并去重
        df = df.sort_values(r_col, ascending=True)
        df = df.drop_duplicates(subset=[w_col], keep='first')
        
        return pd.Series(df[r_col].values, index=df[w_col]).to_dict()
    except: return None

vocab_dict = load_vocab()

# ==========================================
# 4. 界面布局
# ==========================================
st.title("🚀 Vocab Master Pro (Smart Case)")

tab_lemma, tab_grade = st.tabs(["🛠️ 1. 智能还原 (Restore)", "📊 2. 单词分级 (Grade)"])

# ---------------------------------------------------------
# Tab 1: 智能还原
# ---------------------------------------------------------
with tab_lemma:
    c1, c2 = st.columns(2)
    with c1:
        raw_text = st.text_area("输入原始文章", height=400, placeholder="He was excited.\nShe went home.")
        btn_restore = st.button("开始还原", type="primary")
    with c2:
        if btn_restore and raw_text:
            res = smart_lemmatize(raw_text)
            st.code(res, language='text')
            st.caption("👆 点击右上角图标一键复制")
        elif not raw_text:
            st.info("👈 请输入文本")

# ---------------------------------------------------------
# Tab 2: 单词分级 (智能大小写)
# ---------------------------------------------------------
with tab_grade:
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a: current_level = st.number_input("当前水平", 0, 20000, 9000, 500)
    with col_b: target_level = st.number_input("目标水平", 0, 20000, 15000, 500)
    st.divider()
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        input_mode = st.radio("识别模式:", ("自动分词 (Word Mode)", "按行处理 (Phrase Mode)"), horizontal=True)
        grade_input = st.text_area("input_box", height=400, placeholder="ANTI\nSydney\nTable", label_visibility="collapsed")
        btn_grade = st.button("开始分级", type="primary", use_container_width=True)

    with g_col2:
        if not vocab_dict:
            st.error("❌ 词库未加载")
        elif btn_grade and grade_input:
            
            # 1. 获取输入列表
            raw_items = []
            if "按行处理" in input_mode:
                lines = grade_input.split('\n')
                for line in lines:
                    if line.strip(): raw_items.append(line.strip())
            else:
                raw_items = grade_input.split()
            
            # 2. 清洗与大小写处理
            seen = set()
            unique_items = []
            JUNK_WORDS = {'s', 't', 'd', 'm', 'll', 've', 're'}
            
            for item in raw_items:
                item_cleaned = item.strip()
                item_lower = item_cleaned.lower()
                
                # 过滤重复和垃圾词
                if item_lower in seen: continue
                if len(item_lower) < 2 and item_lower not in ['a', 'i']: continue
                if item_lower in JUNK_WORDS: continue
                
                # === 核心修改：智能判断显示的大小写 ===
                display_word = get_display_case(item_cleaned)
                
                seen.add(item_lower)
                unique_items.append(display_word)
            
            # 3. 查词 (统一用小写查)
            data = []
            for item in unique_items:
                lookup_key = item.lower()
                rank = vocab_dict.get(lookup_key, 99999)
                
                cat = "beyond"
                if rank <= current_level: cat = "known"
                elif rank <= target_level: cat = "target"
                
                # data 里存的是 display_word (Sydney / anti)
                data.append({"word": item, "rank": rank, "cat": cat})
            
            # 4. 排序与展示
            df = pd.DataFrame(data)
            if not df.empty:
                df = df.sort_values(by='rank', ascending=True)
                
                t1, t2, t3 = st.tabs([
                    f"🟡 重点 ({len(df[df['cat']=='target'])})", 
                    f"🔴 超纲 ({len(df[df['cat']=='beyond'])})", 
                    f"🟢 已掌握 ({len(df[df['cat']=='known'])})"
                ])
                
                def show(cat_name):
                    sub = df[df['cat'] == cat_name]
                    if sub.empty: 
                        st.info("无")
                    else:
                        txt = "\n".join(sub['word'].tolist())
                        st.code(txt, language='text')
                        st.caption("👆 点击右上角图标一键复制")

                with t1: show("target")
                with t2: show("beyond")
                with t3: show("known")
            else:
                st.warning("无有效单词")